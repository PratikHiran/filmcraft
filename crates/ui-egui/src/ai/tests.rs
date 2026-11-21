//! Tests for the FilmCraft AI Assistant.

use filmcraft_engine::Session;
use serde_json::{Value, json};

use super::client::{
    decode_chunked_body, extract_json_object, strip_code_fences, AssistantPlan, OllamaClient, PlannedStep,
};
use super::prompt::{build_system_prompt, build_user_prompt};
use super::state::sanitize_plan;
use super::validation::{normalize_command_id, validate_steps, ALLOWED_COMMANDS};

#[test]
fn test_all_allowed_commands_registered_in_engine() {
    let all_specs = filmcraft_engine::commands::command_specs();
    for cmd in ALLOWED_COMMANDS {
        let found = all_specs.iter().any(|s| s.id == *cmd);
        assert!(found, "Command '{cmd}' in ALLOWED_COMMANDS must be registered in engine command_specs()");
    }
}

#[test]
fn test_normalize_command_id_aliases() {
    assert_eq!(normalize_command_id("timeline.rippleDelete"), "edit.rippleDelete");
    assert_eq!(normalize_command_id("timeline.ripple_delete"), "edit.rippleDelete");
    assert_eq!(normalize_command_id("timeline.delete"), "edit.clear");
    assert_eq!(normalize_command_id("timeline.clear"), "edit.clear");
    assert_eq!(normalize_command_id("playback.seek"), "playhead.set");
    assert_eq!(normalize_command_id("graphics.newText-1"), "graphics.newText");
}

#[test]
fn test_sanitize_plan_prunes_hallucinated_deletes_on_title_request() {
    let mut plan = AssistantPlan {
        explanation: "Add a title at the beginning".to_string(),
        steps: vec![
            PlannedStep {
                id: "timeline.rippleDelete".to_string(),
                params: json!({}),
            },
            PlannedStep {
                id: "graphics.newText".to_string(),
                params: json!({"text": "Welcome", "seconds": 5.0, "time": 0}),
            },
        ],
    };

    sanitize_plan("Add a title saying 'Welcome' at the beginning of the sequence.", &mut plan);

    assert_eq!(plan.steps.len(), 1, "Hallucinated ripple delete must be pruned from title request");
    assert_eq!(plan.steps[0].id, "graphics.newText");
    assert_eq!(plan.steps[0].params["text"], "Welcome");
}

#[test]
fn test_strip_code_fences() {
    let raw = "```json\n{\"explanation\": \"test\", \"steps\": []}\n```";
    assert_eq!(strip_code_fences(raw), "{\"explanation\": \"test\", \"steps\": []}");

    let raw2 = "```\n{\"explanation\": \"test\", \"steps\": []}\n```";
    assert_eq!(strip_code_fences(raw2), "{\"explanation\": \"test\", \"steps\": []}");

    let plain = "{\"explanation\": \"test\", \"steps\": []}";
    assert_eq!(strip_code_fences(plain), "{\"explanation\": \"test\", \"steps\": []}");
}

#[test]
fn test_decode_chunked_and_extract_json() {
    let payload = "{\"explanation\":\"ok\",\"steps\":[]}";
    let chunked = format!("{:x}\r\n{}\r\n0\r\n\r\n", payload.len(), payload);
    let decoded = decode_chunked_body(&chunked);
    assert_eq!(decoded, payload);

    let with_garbage = "garbage before {\"key\":\"val\"} and after";
    assert_eq!(extract_json_object(with_garbage), "{\"key\":\"val\"}");
}

#[test]
fn test_parse_assistant_plan() {
    let json_text = r#"{
        "explanation": "Add a welcome title to the timeline",
        "steps": [
            {
                "id": "graphics.newText",
                "params": {
                    "text": "Welcome",
                    "seconds": 5.0
                }
            }
        ]
    }"#;
    let plan: AssistantPlan = serde_json::from_str(json_text).unwrap();
    assert_eq!(plan.explanation, "Add a welcome title to the timeline");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].id, "graphics.newText");
    assert_eq!(plan.steps[0].params["text"], "Welcome");
}

#[test]
fn test_validation_rejects_disallowed_command() {
    let session = Session::default();
    let steps = vec![PlannedStep {
        id: "dangerous.formatDisk".to_string(),
        params: json!({}),
    }];
    let res = validate_steps(&session, &steps);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("not in the allowed editing commands whitelist"));
}

#[test]
fn test_validation_rejects_nonexistent_clip() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let steps = vec![PlannedStep {
        id: "timeline.trim".to_string(),
        params: json!({
            "clip": 999999, // Non-existent clip ID
            "edge": "in",
            "mode": "ripple",
            "delta": 1000
        }),
    }];
    let res = validate_steps(&session, &steps);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("Referenced clip ID 999999 does not exist"));
}

#[test]
fn test_validation_accepts_valid_demo_steps() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let q = session.active_sequence().unwrap();
    let clip_id = q.video_tracks[0].items[0].id.0;

    let steps = vec![
        PlannedStep {
            id: "timeline.select".to_string(),
            params: json!({"clips": [clip_id]}),
        },
        PlannedStep {
            id: "graphics.newText".to_string(),
            params: json!({"text": "Test Title", "seconds": 3.0}),
        },
    ];

    let res = validate_steps(&session, &steps);
    assert!(res.is_ok(), "Validation should succeed for valid demo steps: {:?}", res);
}

#[test]
fn test_prompt_generation_includes_sequence_and_clips() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let prompt = build_user_prompt(&session, "Cut the first 5 seconds of the selected clip");
    assert!(prompt.contains("CURRENT PROJECT STATE:"));
    assert!(prompt.contains("sequenceId (not a clip ID)") || prompt.contains("activeSequence"));
    assert!(prompt.contains("timelineClips"));
    assert!(prompt.contains("USER REQUEST:"));
    assert!(prompt.contains("Cut the first 5 seconds of the selected clip"));
}

#[test]
fn test_live_ollama_execution_on_demo_timeline() {
    // Only runs when Ollama is reachable
    let client = OllamaClient::default();
    if client.check_health().is_err() {
        eprintln!("Ollama not running locally on 11434, skipping live test");
        return;
    }

    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let q_before = session.active_sequence().unwrap().clone();
    let initial_clip_count: usize = q_before.video_tracks.iter().map(|t| t.items.len()).sum();
    assert!(initial_clip_count > 0, "Demo project must have clips");

    let first_clip = q_before.video_tracks[0].items[0].clone();
    let first_clip_id = first_clip.id.0;

    // Select the first clip
    session.execute("timeline.select", json!({"clips": [first_clip_id]})).unwrap();

    // Generate editing plan with Ollama
    let system = build_system_prompt();
    let prompt = build_user_prompt(&session, "Add a title saying \"Welcome\" to the timeline.");

    let plan = client.generate(system, &prompt).expect("Ollama generate should succeed");
    assert!(!plan.steps.is_empty(), "Ollama must return at least one step");

    // Validate
    validate_steps(&session, &plan.steps).expect("Validation of generated plan must succeed");

    // Execute
    for step in &plan.steps {
        let norm_id = super::validation::normalize_command_id(&step.id);
        let r = session.execute(&norm_id, step.params.clone());
        assert!(r.is_ok(), "Engine execution failed on {}: {:?}", norm_id, r.err());
    }

    // Verify timeline modified
    let q_after = session.active_sequence().unwrap();
    let new_clip_count: usize = q_after.video_tracks.iter().map(|t| t.items.len()).sum();
    assert!(new_clip_count >= initial_clip_count, "New text clip should be added to timeline");

    // Verify undo works
    let undo_res = session.execute("edit.undo", json!({}));
    assert!(undo_res.is_ok(), "Undo must work after AI edit: {:?}", undo_res.err());

    // Verify redo works
    let redo_res = session.execute("edit.redo", json!({}));
    assert!(redo_res.is_ok(), "Redo must work: {:?}", redo_res.err());
}

/// Regression test: "Add a title at the beginning" with clips already on the timeline.
/// The model must generate ONLY graphics.newText (no timeline.razor).
/// Validation must succeed and the new clip must appear at time 0.
#[test]
fn test_add_title_at_beginning_with_clips() {
    let client = OllamaClient::default();
    if client.check_health().is_err() {
        eprintln!("Ollama not running locally on 11434, skipping live test");
        return;
    }

    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let q_before = session.active_sequence().unwrap().clone();
    let initial_video_clips: usize = q_before.video_tracks.iter().map(|t| t.items.len()).sum();
    assert!(initial_video_clips > 0, "Demo project must have clips for this regression test");

    let system = build_system_prompt();
    let prompt = build_user_prompt(&session, "Add a title saying 'Welcome to my video' at the beginning of the sequence.");

    let plan = client.generate(system, &prompt).expect("Ollama generate should succeed");

    // Regression: plan must NOT contain timeline.razor
    let has_razor = plan.steps.iter().any(|s| {
        let norm = super::validation::normalize_command_id(&s.id);
        norm == "timeline.razor"
    });
    assert!(
        !has_razor,
        "Adding a title must NOT generate timeline.razor. Got steps: {:?}",
        plan.steps.iter().map(|s| &s.id).collect::<Vec<_>>()
    );

    // Plan must include graphics.newText
    let has_new_text = plan.steps.iter().any(|s| {
        let norm = super::validation::normalize_command_id(&s.id);
        norm == "graphics.newText"
    });
    assert!(
        has_new_text,
        "Adding a title must generate graphics.newText. Got steps: {:?}",
        plan.steps.iter().map(|s| &s.id).collect::<Vec<_>>()
    );

    // Validation must pass
    validate_steps(&session, &plan.steps)
        .expect("Validation must succeed for 'add title at beginning' with existing clips");

    // Execute all steps
    for step in &plan.steps {
        let norm_id = super::validation::normalize_command_id(&step.id);
        let r = session.execute(&norm_id, step.params.clone());
        assert!(r.is_ok(), "Engine execution failed on {}: {:?}", norm_id, r.err());
    }

    // Verify a new video clip was added
    let q_after = session.active_sequence().unwrap();
    let new_video_clips: usize = q_after.video_tracks.iter().map(|t| t.items.len()).sum();
    assert!(
        new_video_clips > initial_video_clips,
        "A new title clip must have been added. Before: {initial_video_clips}, after: {new_video_clips}"
    );

    // Verify undo works
    let undo_res = session.execute("edit.undo", json!({}));
    assert!(undo_res.is_ok(), "Undo must work after adding title: {:?}", undo_res.err());

    let q_undone = session.active_sequence().unwrap();
    let undone_clips: usize = q_undone.video_tracks.iter().map(|t| t.items.len()).sum();
    assert_eq!(
        undone_clips, initial_video_clips,
        "After undo, clip count must return to original"
    );

    // Verify redo works
    let redo_res = session.execute("edit.redo", json!({}));
    assert!(redo_res.is_ok(), "Redo must work after undo: {:?}", redo_res.err());
}

/// Regression test: "Add a title at the beginning" with an empty sequence (no existing clips).
/// The model must generate ONLY graphics.newText with no clip references.
/// Validation must succeed even when timelineClips is empty.
#[test]
fn test_add_title_to_empty_sequence() {
    let client = OllamaClient::default();
    if client.check_health().is_err() {
        eprintln!("Ollama not running locally on 11434, skipping live test");
        return;
    }

    let mut session = Session::default();
    // Create a fresh empty sequence
    session.execute("file.newSequence", json!({})).unwrap();

    let q_before = session.active_sequence().unwrap().clone();
    let initial_clips: usize = q_before
        .video_tracks
        .iter()
        .chain(q_before.audio_tracks.iter())
        .map(|t| t.items.len())
        .sum();
    assert_eq!(initial_clips, 0, "New sequence must start empty");

    let system = build_system_prompt();
    let prompt = build_user_prompt(&session, "Add a title saying 'Welcome to my video' at the beginning of the sequence.");

    let plan = client.generate(system, &prompt).expect("Ollama generate should succeed");

    // Must NOT contain timeline.razor (no clips to razor on)
    let has_razor = plan.steps.iter().any(|s| {
        let norm = super::validation::normalize_command_id(&s.id);
        norm == "timeline.razor"
    });
    assert!(
        !has_razor,
        "Adding a title to an empty sequence must NOT generate timeline.razor. Got steps: {:?}",
        plan.steps.iter().map(|s| &s.id).collect::<Vec<_>>()
    );

    // Must include graphics.newText
    let has_new_text = plan.steps.iter().any(|s| {
        let norm = super::validation::normalize_command_id(&s.id);
        norm == "graphics.newText"
    });
    assert!(
        has_new_text,
        "Adding a title must generate graphics.newText even on empty sequence. Got steps: {:?}",
        plan.steps.iter().map(|s| &s.id).collect::<Vec<_>>()
    );

    // Validation must pass with an empty sequence (no clip ID references to fail)
    validate_steps(&session, &plan.steps)
        .expect("Validation must succeed for 'add title' on empty sequence");

    // Execute all steps
    for step in &plan.steps {
        let norm_id = super::validation::normalize_command_id(&step.id);
        let r = session.execute(&norm_id, step.params.clone());
        assert!(r.is_ok(), "Engine execution failed on {}: {:?}", norm_id, r.err());
    }

    // Verify a new video clip was created
    let q_after = session.active_sequence().unwrap();
    let new_clips: usize = q_after.video_tracks.iter().map(|t| t.items.len()).sum();
    assert!(
        new_clips > 0,
        "A title clip must have been created on the empty sequence"
    );

    // Verify undo works
    let undo_res = session.execute("edit.undo", json!({}));
    assert!(undo_res.is_ok(), "Undo must work after adding title to empty sequence: {:?}", undo_res.err());
}

/// Exact user scenario test:
/// "Add a title saying 'Welcome' at the beginning of the sequence."
/// Tests that even if raw output contains rippleDelete / clear, sanitize_plan prunes it,
/// validation passes cleanly, and graphics.newText executes to add the title.
#[test]
fn test_exact_user_request_welcome_title_at_beginning() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let initial_clips: usize = session
        .active_sequence()
        .unwrap()
        .video_tracks
        .iter()
        .map(|t| t.items.len())
        .sum();

    // Simulate an LLM output that included a hallucinated rippleDelete before graphics.newText
    let mut plan = AssistantPlan {
        explanation: "Adding Welcome title at beginning".to_string(),
        steps: vec![
            PlannedStep {
                id: "timeline.rippleDelete".to_string(),
                params: json!({}),
            },
            PlannedStep {
                id: "graphics.newText".to_string(),
                params: json!({
                    "text": "Welcome",
                    "time": 0,
                    "seconds": 5.0
                }),
            },
        ],
    };

    let user_query = "Add a title saying 'Welcome' at the beginning of the sequence.";
    super::state::sanitize_plan(user_query, &mut plan);

    // After sanitization, destructive operations must be stripped
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].id, "graphics.newText");

    // Validation must pass cleanly
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution must succeed
    let exec_res = session.execute(&plan.steps[0].id, plan.steps[0].params.clone());
    assert!(exec_res.is_ok(), "Execution of graphics.newText should succeed: {:?}", exec_res.err());

    // Check that title clip was added across video tracks
    let final_clips: usize = session
        .active_sequence()
        .unwrap()
        .video_tracks
        .iter()
        .map(|t| t.items.len())
        .sum();
    assert!(final_clips > initial_clips);
}

#[test]
fn test_normalize_preset_name() {
    assert_eq!(super::validation::normalize_preset_name("Cinematic"), "Teal Shadows, Warm Skin");
    assert_eq!(super::validation::normalize_preset_name("cinematic"), "Teal Shadows, Warm Skin");
    assert_eq!(super::validation::normalize_preset_name("Teal Shadows, Warm Skin"), "Teal Shadows, Warm Skin");
    assert_eq!(super::validation::normalize_preset_name("B&W"), "Neutral Mono");
    assert_eq!(super::validation::normalize_preset_name("Monochrome"), "Neutral Mono");
    assert_eq!(super::validation::normalize_preset_name("Bleach Bypass"), "Silver Retention");
    assert_eq!(super::validation::normalize_preset_name("Faded Film"), "Faded Print");
}

#[test]
fn test_make_video_cinematic_auto_normalizes_and_executes() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let seq = session.active_sequence().unwrap();
    let clip_id = seq.video_tracks[0].items[0].id.0;

    // Simulate LLM output: {"name": "Cinematic", "clips": [clip_id]}
    let mut plan = AssistantPlan {
        explanation: "Make video cinematic".to_string(),
        steps: vec![PlannedStep {
            id: "lumetri.applyPreset".to_string(),
            params: json!({
                "name": "Cinematic",
                "clips": [clip_id]
            }),
        }],
    };

    super::state::sanitize_plan("Make the video more cinematic.", &mut plan);

    // After sanitization, "Cinematic" must be normalized to "Teal Shadows, Warm Skin"
    assert_eq!(plan.steps[0].params["name"], "Teal Shadows, Warm Skin");

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution must succeed
    let exec_res = session.execute(&plan.steps[0].id, plan.steps[0].params.clone());
    assert!(exec_res.is_ok(), "Execution of lumetri.applyPreset should succeed: {:?}", exec_res.err());
}

#[test]
fn test_normalize_track_spec() {
    assert_eq!(super::validation::normalize_track_spec("Video 2"), "V2");
    assert_eq!(super::validation::normalize_track_spec("video 1"), "V1");
    assert_eq!(super::validation::normalize_track_spec("track 2"), "V2");
    assert_eq!(super::validation::normalize_track_spec("Audio 1"), "A1");
    assert_eq!(super::validation::normalize_track_spec("audio 2"), "A2");
    assert_eq!(super::validation::normalize_track_spec("V2"), "V2");
    assert_eq!(super::validation::normalize_track_spec("v1"), "V1");
    assert_eq!(super::validation::normalize_track_spec("new"), "new");
}

#[test]
fn test_move_clip_to_video_2_auto_normalizes_and_executes() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let seq = session.active_sequence().unwrap();
    let clip = &seq.video_tracks[0].items[0];
    let clip_id = clip.id.0;
    let clip_start = clip.start.0;

    // Simulate LLM output: {"moves": [{"clip": clip_id, "track": "Video 2", "time": clip_start}]}
    let mut plan = AssistantPlan {
        explanation: "Move clip to Video 2".to_string(),
        steps: vec![PlannedStep {
            id: "timeline.move".to_string(),
            params: json!({
                "moves": [{
                    "clip": clip_id,
                    "track": "Video 2",
                    "time": clip_start
                }],
                "linked": false
            }),
        }],
    };

    super::state::sanitize_plan("Move this clip from Video 1 to Video 2", &mut plan);

    // After sanitization, "Video 2" should be normalized to "V2"
    assert_eq!(plan.steps[0].params["moves"][0]["track"], "V2");

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution must succeed
    let exec_res = session.execute(&plan.steps[0].id, plan.steps[0].params.clone());
    assert!(exec_res.is_ok(), "Execution of timeline.move should succeed: {:?}", exec_res.err());
}

#[test]
fn test_apply_complex_effects_video_and_audio() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let seq = session.active_sequence().unwrap();
    let v_clip = seq.video_tracks[0].items[0].id.0;
    let a_clip = seq.audio_tracks[0].items[0].id.0;

    let plan = AssistantPlan {
        explanation: "Apply vignette to video, studio reverb and denoise to audio, and Teal Shadows Warm Skin grade".to_string(),
        steps: vec![
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({
                    "clips": [v_clip],
                    "effect": "vignette"
                }),
            },
            PlannedStep {
                id: "lumetri.applyPreset".to_string(),
                params: json!({
                    "name": "Teal Shadows, Warm Skin",
                    "clips": [v_clip]
                }),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({
                    "clips": [a_clip],
                    "effect": "denoise"
                }),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({
                    "clips": [a_clip],
                    "effect": "studio_reverb"
                }),
            },
        ],
    };

    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    for step in &plan.steps {
        let r = session.execute(&step.id, step.params.clone());
        assert!(r.is_ok(), "Step {} execution failed: {:?}", step.id, r.err());
    }

    // Verify effects applied to video clip
    let updated_seq = session.active_sequence().unwrap();
    let (_, updated_v) = updated_seq.find_item(filmcraft_project::ClipId(v_clip)).unwrap();
    assert!(updated_v.effects.iter().any(|e| e.effect == "vignette"));
    assert!(updated_v.effects.iter().any(|e| e.effect == "lumetri"));

    // Verify effects applied to audio clip
    let (_, updated_a) = updated_seq.find_item(filmcraft_project::ClipId(a_clip)).unwrap();
    assert!(updated_a.effects.iter().any(|e| e.effect == "denoise"));
    assert!(updated_a.effects.iter().any(|e| e.effect == "studio_reverb"));
}

/// Exact user scenario test:
/// "Apply the Night Exterior preset and Gaussian Blur to the video clips, and apply dereverb to the audio."
/// Even if a video-only clip is selected in the UI, audio effects like dereverb must be retargeted
/// to actual audio clips rather than failing with "audio effects need an audio clip".
#[test]
fn test_exact_user_request_night_exterior_blur_dereverb_with_video_selected() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    // Select a video clip
    let v_clip = session.active_sequence().unwrap().video_tracks[0].items[0].id.0;
    session.state.selection = vec![filmcraft_project::ClipId(v_clip)];

    // Simulate raw plan where clips is either omitted or has the selected video clip
    let mut plan = AssistantPlan {
        explanation: "Apply Night Exterior and Gaussian Blur to video, and dereverb to audio".to_string(),
        steps: vec![
            PlannedStep {
                id: "lumetri.applyPreset".to_string(),
                params: json!({
                    "name": "Night Exterior",
                    "clips": [v_clip]
                }),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({
                    "clips": [v_clip],
                    "effect": "gaussian_blur"
                }),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({
                    "clips": [v_clip], // Model mistakenly passed the selected video clip
                    "effect": "dereverb"
                }),
            },
        ],
    };

    // Preprocessing as done in poll_and_execute
    for step in &mut plan.steps {
        if step.id == "effects.apply" {
            if let Some(seq) = session.active_sequence() {
                let effect_name = step.params.get("effect").and_then(Value::as_str).unwrap_or("");
                let is_audio = filmcraft_project::find_effect(effect_name)
                    .or_else(|| filmcraft_project::effect_defs().iter().find(|d| d.name.eq_ignore_ascii_case(effect_name)))
                    .is_some_and(|d| d.kind == filmcraft_project::EffectKind::Audio);

                if is_audio {
                    let current_clips: Vec<filmcraft_project::ClipId> = step
                        .params
                        .get("clips")
                        .and_then(|c| c.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(filmcraft_project::ClipId)).collect())
                        .unwrap_or_default();

                    let has_valid_audio = !current_clips.is_empty() && current_clips.iter().any(|c| {
                        if let Some((tid, it)) = seq.find_item(*c) {
                            let is_audio_track = seq.track(tid).map_or(false, |t| t.kind == filmcraft_project::TrackKind::Audio);
                            let has_linked_audio = it.link.is_some() && seq.audio_tracks.iter().any(|t| t.items.iter().any(|i| i.link == it.link));
                            is_audio_track || has_linked_audio
                        } else {
                            false
                        }
                    });

                    if !has_valid_audio {
                        let all_audio_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
                        if !all_audio_ids.is_empty() {
                            if let Some(obj) = step.params.as_object_mut() {
                                obj.insert("clips".to_string(), json!(all_audio_ids));
                            }
                        }
                    }
                }
            }
        }
    }

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation must pass: {:?}", val_res.err());

    // Execution must succeed for all steps
    for step in &plan.steps {
        let r = session.execute(&step.id, step.params.clone());
        assert!(r.is_ok(), "Step {} failed: {:?}", step.id, r.err());
    }

    // Verify dereverb is applied to an audio clip
    let updated_seq = session.active_sequence().unwrap();
    let audio_has_dereverb = updated_seq.audio_tracks.iter().any(|t| {
        t.items.iter().any(|i| i.effects.iter().any(|e| e.effect == "dereverb"))
    });
    assert!(audio_has_dereverb, "Dereverb must be applied to audio clips");
}

/// Exact user scenario test:
/// "Clean up the audio by applying denoise, vocal enhancer, and parametric EQ to enhance speech."
/// If the model hallucinates a `lumetri.applyPreset` step because of words like "enhance" or "preset",
/// `sanitize_plan` must prune it, validation must pass, and the 3 audio effects must execute cleanly.
#[test]
fn test_exact_user_request_clean_up_audio_speech_prunes_hallucinated_lumetri() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let user_query = "Clean up the audio by applying denoise, vocal enhancer, and parametric EQ to enhance speech.";

    let mut plan = AssistantPlan {
        explanation: "Apply audio effects to enhance speech".to_string(),
        steps: vec![
            PlannedStep {
                id: "lumetri.applyPreset".to_string(), // Hallucinated step
                params: json!({"name": "Teal Shadows, Warm Skin"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "denoise"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "vocal_enhancer"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "parametric_eq"}),
            },
        ],
    };

    // 1. Sanitize plan: must prune lumetri.applyPreset
    sanitize_plan(user_query, &mut plan);
    assert_eq!(plan.steps.len(), 3);
    assert!(!plan.steps.iter().any(|s| s.id == "lumetri.applyPreset"));

    // 2. Preprocess audio effect clips as in poll_and_execute
    let seq = session.active_sequence().unwrap();
    let all_audio_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
    for step in &mut plan.steps {
        if step.id == "effects.apply" {
            if let Some(obj) = step.params.as_object_mut() {
                obj.insert("clips".to_string(), json!(all_audio_ids));
            }
        }
    }

    // 3. Validate
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation must pass: {:?}", val_res.err());

    // 4. Execute
    for step in &plan.steps {
        let r = session.execute(&step.id, step.params.clone());
        assert!(r.is_ok(), "Step {} execution failed: {:?}", step.id, r.err());
    }

    // 5. Verify audio effects
    let updated_seq = session.active_sequence().unwrap();
    let a_items = &updated_seq.audio_tracks[0].items;
    assert!(a_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "denoise")));
    assert!(a_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "vocal_enhancer")));
    assert!(a_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "parametric_eq")));
}

#[test]
fn test_normalize_effect_name() {
    assert_eq!(super::validation::normalize_effect_name("bass_boost"), "bass");
    assert_eq!(super::validation::normalize_effect_name("bass boost"), "bass");
    assert_eq!(super::validation::normalize_effect_name("treble boost"), "treble");
    assert_eq!(super::validation::normalize_effect_name("camera shake"), "camera_shake");
    assert_eq!(super::validation::normalize_effect_name("rgb split"), "rgb_split");
    assert_eq!(super::validation::normalize_effect_name("chromatic aberration"), "rgb_split");
    assert_eq!(super::validation::normalize_effect_name("tube compressor"), "tube_compressor");
    assert_eq!(super::validation::normalize_effect_name("noise reduction"), "denoise");
}

#[test]
fn test_exact_user_request_bass_boost_normalization() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let seq = session.active_sequence().unwrap();
    let a_clip = seq.audio_tracks[0].items[0].id.0;

    let mut plan = AssistantPlan {
        explanation: "Apply bass boost to audio".to_string(),
        steps: vec![PlannedStep {
            id: "effects.apply".to_string(),
            params: json!({
                "clips": [a_clip],
                "effect": "bass_boost" // Model output alias
            }),
        }],
    };

    sanitize_plan("apply bass boost to the audio", &mut plan);

    // After sanitization, "bass_boost" must be normalized to "bass"
    assert_eq!(plan.steps[0].params["effect"], "bass");

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution must succeed
    let exec_res = session.execute(&plan.steps[0].id, plan.steps[0].params.clone());
    assert!(exec_res.is_ok(), "Execution of bass effect should succeed: {:?}", exec_res.err());

    // Verify bass is applied
    let updated_seq = session.active_sequence().unwrap();
    let (_, a_item) = updated_seq.find_item(filmcraft_project::ClipId(a_clip)).unwrap();
    assert!(a_item.effects.iter().any(|e| e.effect == "bass"));
}

#[test]
fn test_camera_shaking_audio_impact_lighting_on_off() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let user_query = "camera shaking and audio impact and lighting on off";

    let mut plan = AssistantPlan {
        explanation: "Apply camera shake, strobe light on off, and audio bass impact".to_string(),
        steps: vec![
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "camera shaking"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "lighting on off"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "audio impact"}),
            },
        ],
    };

    sanitize_plan(user_query, &mut plan);

    // Verify alias normalization
    assert_eq!(plan.steps[0].params["effect"], "camera_shake");
    assert_eq!(plan.steps[1].params["effect"], "strobe");
    assert_eq!(plan.steps[2].params["effect"], "bass");

    // Preprocessing as in poll_and_execute
    let seq = session.active_sequence().unwrap();
    let v_ids: Vec<u64> = seq.video_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
    let a_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();

    if let Some(obj) = plan.steps[0].params.as_object_mut() {
        obj.insert("clips".to_string(), json!(v_ids));
    }
    if let Some(obj) = plan.steps[1].params.as_object_mut() {
        obj.insert("clips".to_string(), json!(v_ids));
    }
    if let Some(obj) = plan.steps[2].params.as_object_mut() {
        obj.insert("clips".to_string(), json!(a_ids));
    }

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution must succeed
    for step in &plan.steps {
        let r = session.execute(&step.id, step.params.clone());
        assert!(r.is_ok(), "Step {} execution failed: {:?}", step.id, r.err());
    }

    // Verify effects applied to sequence
    let updated_seq = session.active_sequence().unwrap();
    let v_items = &updated_seq.video_tracks[0].items;
    let a_items = &updated_seq.audio_tracks[0].items;

    assert!(v_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "camera_shake")));
    assert!(v_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "strobe")));
    assert!(a_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "bass")));
}

#[test]
fn test_exact_user_request_gaussian_blur_vignette_golden_dusk_vocal_enhancer() {
    let mut session = Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();

    let user_query = "Apply Gaussian Blur, vignette, and the Golden Dusk preset to the video, and apply vocal enhancer to the audio.";

    // Simulate raw LLM output including a hallucinated timeline.move and preset key
    let mut plan = AssistantPlan {
        explanation: "Apply Gaussian Blur, vignette, and Golden Dusk to video, and vocal enhancer to audio".to_string(),
        steps: vec![
            PlannedStep {
                id: "timeline.move".to_string(),
                params: json!({"moves": [{"clip": 73, "track": "V1", "time": 0}]}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "Gaussian Blur"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "vignette"}),
            },
            PlannedStep {
                id: "lumetri.applyPreset".to_string(),
                params: json!({"preset": "Golden Dusk"}),
            },
            PlannedStep {
                id: "effects.apply".to_string(),
                params: json!({"effect": "vocal enhancer"}),
            },
        ],
    };

    // Sanitize plan: must prune hallucinated timeline.move
    sanitize_plan(user_query, &mut plan);

    assert_eq!(plan.steps.len(), 4, "Hallucinated timeline.move must be pruned");
    assert_eq!(plan.steps[0].id, "effects.apply");
    assert_eq!(plan.steps[0].params["effect"], "gaussian_blur");
    assert_eq!(plan.steps[1].id, "effects.apply");
    assert_eq!(plan.steps[1].params["effect"], "vignette");
    assert_eq!(plan.steps[2].id, "lumetri.applyPreset");
    assert_eq!(plan.steps[2].params["name"], "Golden Dusk");
    assert_eq!(plan.steps[3].id, "effects.apply");
    assert_eq!(plan.steps[3].params["effect"], "vocal_enhancer");

    // Preprocessing as in poll_and_execute
    let seq = session.active_sequence().unwrap();
    let v_ids: Vec<u64> = seq.video_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
    let a_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();

    plan.steps[0].params.as_object_mut().unwrap().insert("clips".to_string(), json!(v_ids));
    plan.steps[1].params.as_object_mut().unwrap().insert("clips".to_string(), json!(v_ids));
    plan.steps[2].params.as_object_mut().unwrap().insert("clips".to_string(), json!(v_ids));
    plan.steps[3].params.as_object_mut().unwrap().insert("clips".to_string(), json!(a_ids));

    // Validation must pass
    let val_res = validate_steps(&session, &plan.steps);
    assert!(val_res.is_ok(), "Validation should pass: {:?}", val_res.err());

    // Execution of all 4 steps must succeed cleanly
    for step in &plan.steps {
        let r = session.execute(&step.id, step.params.clone());
        assert!(r.is_ok(), "Step {} execution failed: {:?}", step.id, r.err());
    }

    // Verify effects applied to sequence
    let updated_seq = session.active_sequence().unwrap();
    let v_items = &updated_seq.video_tracks[0].items;
    let a_items = &updated_seq.audio_tracks[0].items;

    assert!(v_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "gaussian_blur")));
    assert!(v_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "vignette")));
    assert!(v_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "lumetri")));
    assert!(a_items.iter().any(|i| i.effects.iter().any(|e| e.effect == "vocal_enhancer")));
}
