//! State management, threading, and execution lifecycle for the FilmCraft AI Assistant.

use std::sync::mpsc::{Receiver, Sender, channel};
use filmcraft_engine::Session;
use serde_json::{Value, json};

use super::client::{AssistantPlan, OllamaClient};
use super::prompt::{build_system_prompt, build_user_prompt};
use super::validation::validate_steps;

/// Record of an executed step.
#[derive(Debug, Clone)]
pub struct StepExecution {
    pub id: String,
    pub params: Value,
    pub success: bool,
    pub detail: String,
}

/// One round of interaction between user and assistant.
#[derive(Debug, Clone)]
pub struct AssistantExchange {
    pub user_query: String,
    pub explanation: String,
    pub steps: Vec<StepExecution>,
    pub overall_error: Option<String>,
}

/// Request sent to the background Ollama worker thread.
pub struct WorkerRequest {
    pub system: String,
    pub prompt: String,
    pub user_query: String,
    pub endpoint: String,
    pub model: String,
}

/// Response returned from the background worker.
pub struct WorkerResponse {
    pub user_query: String,
    pub result: Result<AssistantPlan, String>,
}

/// State of the AI Editing Assistant.
pub struct AssistantState {
    pub prompt_input: String,
    pub is_processing: bool,
    pub status_message: Option<String>,
    pub error_message: Option<String>,
    pub history: Vec<AssistantExchange>,
    pub open: bool,
    pub endpoint: String,
    pub model: String,
    worker_tx: Sender<WorkerRequest>,
    result_rx: Receiver<WorkerResponse>,
}

impl Default for AssistantState {
    fn default() -> Self {
        Self::new()
    }
}

impl AssistantState {
    pub fn new() -> Self {
        let (worker_tx, worker_rx) = channel::<WorkerRequest>();
        let (result_tx, result_rx) = channel::<WorkerResponse>();

        // Background worker thread for local LLM inference
        let _ = std::thread::Builder::new()
            .name("filmcraft-ai-worker".to_string())
            .spawn(move || {
                while let Ok(req) = worker_rx.recv() {
                    let client = OllamaClient::new(req.endpoint, req.model);
                    let result = client.generate(&req.system, &req.prompt);
                    let _ = result_tx.send(WorkerResponse {
                        user_query: req.user_query,
                        result,
                    });
                }
            });

        Self {
            prompt_input: String::new(),
            is_processing: false,
            status_message: None,
            error_message: None,
            history: Vec::new(),
            open: true, // Open by default for immediate visibility
            endpoint: "http://localhost:11434".to_string(),
            model: "qwen2.5:3b".to_string(),
            worker_tx,
            result_rx,
        }
    }

    /// Submit a user prompt to the background worker.
    pub fn submit_prompt(&mut self, session: &Session) {
        let query = self.prompt_input.trim().to_string();
        if query.is_empty() || self.is_processing {
            return;
        }

        self.is_processing = true;
        self.status_message = Some(format!("Consulting local {}...", self.model));
        self.error_message = None;

        let system = build_system_prompt().to_string();
        let prompt = build_user_prompt(session, &query);

        let req = WorkerRequest {
            system,
            prompt,
            user_query: query,
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
        };

        if let Err(e) = self.worker_tx.send(req) {
            self.is_processing = false;
            self.error_message = Some(format!("Failed to submit request to background worker: {e}"));
        } else {
            self.prompt_input.clear();
        }
    }

    /// Submit a predefined prompt directly.
    pub fn submit_preset(&mut self, session: &Session, prompt: &str) {
        self.prompt_input = prompt.to_string();
        self.submit_prompt(session);
    }

    /// Poll for results from the background worker and execute valid commands on the UI thread.
    pub fn poll_and_execute(&mut self, session: &mut Session) {
        let Ok(resp) = self.result_rx.try_recv() else {
            return;
        };

        self.is_processing = false;

        match resp.result {
            Err(e) => {
                self.error_message = Some(e.clone());
                self.status_message = None;
                self.history.push(AssistantExchange {
                    user_query: resp.user_query,
                    explanation: String::new(),
                    steps: Vec::new(),
                    overall_error: Some(e),
                });
            }
            Ok(mut plan) => {
                // Sanitize and align planned steps with user intent
                sanitize_plan(&resp.user_query, &mut plan);

                // Auto-populate and validate clips for lumetri.applyPreset
                for step in &mut plan.steps {
                    if step.id == "lumetri.applyPreset" {
                        if let Some(seq) = session.active_sequence() {
                            // Filter out any audio clips from lumetri.applyPreset
                            if let Some(clips_arr) = step.params.get_mut("clips").and_then(Value::as_array_mut) {
                                clips_arr.retain(|c| {
                                    if let Some(cid) = c.as_u64() {
                                        seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id.0 == cid))
                                    } else {
                                        false
                                    }
                                });
                            }
                            let has_clips = step
                                .params
                                .get("clips")
                                .and_then(|c| c.as_array())
                                .map_or(false, |a| !a.is_empty());
                            if !has_clips {
                                let sel_video: Vec<u64> = session.state.selection.iter().filter(|c| {
                                    seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id == **c))
                                }).map(|c| c.0).collect();

                                if !sel_video.is_empty() {
                                    if let Some(obj) = step.params.as_object_mut() {
                                        obj.insert("clips".to_string(), json!(sel_video));
                                    }
                                } else {
                                    let all_video_ids: Vec<u64> = seq
                                        .video_tracks
                                        .iter()
                                        .flat_map(|t| t.items.iter().map(|i| i.id.0))
                                        .collect();
                                    if !all_video_ids.is_empty() {
                                        if let Some(obj) = step.params.as_object_mut() {
                                            obj.insert("clips".to_string(), json!(all_video_ids));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Auto-populate clips for effects.apply if not provided or incorrectly targeted
                    if step.id == "effects.apply" {
                        if let Some(seq) = session.active_sequence() {
                            // Normalize effect name/alias first (e.g. "bass_boost" -> "bass")
                            if let Some(obj) = step.params.as_object_mut() {
                                if let Some(effect_val) = obj.get("effect").and_then(Value::as_str) {
                                    let norm_effect = super::validation::normalize_effect_name(effect_val);
                                    obj.insert("effect".to_string(), json!(norm_effect));
                                }
                            }
                            let effect_name = step.params.get("effect").and_then(Value::as_str).unwrap_or("");
                            let is_audio = filmcraft_project::find_effect(effect_name)
                                .or_else(|| filmcraft_project::effect_defs().iter().find(|d| d.name.eq_ignore_ascii_case(effect_name)))
                                .is_some_and(|d| d.kind == filmcraft_project::EffectKind::Audio);

                            if is_audio {
                                // Check if currently specified clips contain audio or linked audio
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

                                // If no valid audio target was found, retarget to audio tracks!
                                if !has_valid_audio {
                                    let sel_audio: Vec<u64> = session.state.selection.iter().filter(|c| {
                                        if let Some((tid, it)) = seq.find_item(**c) {
                                            let is_audio_track = seq.track(tid).map_or(false, |t| t.kind == filmcraft_project::TrackKind::Audio);
                                            let has_linked_audio = it.link.is_some() && seq.audio_tracks.iter().any(|t| t.items.iter().any(|i| i.link == it.link));
                                            is_audio_track || has_linked_audio
                                        } else {
                                            false
                                        }
                                    }).map(|c| c.0).collect();

                                    if !sel_audio.is_empty() {
                                        if let Some(obj) = step.params.as_object_mut() {
                                            obj.insert("clips".to_string(), json!(sel_audio));
                                        }
                                    } else {
                                        let all_audio_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
                                        if !all_audio_ids.is_empty() {
                                            if let Some(obj) = step.params.as_object_mut() {
                                                obj.insert("clips".to_string(), json!(all_audio_ids));
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Video effect: ensure it targets valid video clips
                                if let Some(clips_arr) = step.params.get_mut("clips").and_then(Value::as_array_mut) {
                                    clips_arr.retain(|c| {
                                        if let Some(cid) = c.as_u64() {
                                            seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id.0 == cid))
                                        } else {
                                            false
                                        }
                                    });
                                }
                                let has_clips = step
                                    .params
                                    .get("clips")
                                    .and_then(|c| c.as_array())
                                    .map_or(false, |a| !a.is_empty());
                                if !has_clips {
                                    let sel_video: Vec<u64> = session.state.selection.iter().filter(|c| {
                                        seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id == **c))
                                    }).map(|c| c.0).collect();

                                    if !sel_video.is_empty() {
                                        if let Some(obj) = step.params.as_object_mut() {
                                            obj.insert("clips".to_string(), json!(sel_video));
                                        }
                                    } else {
                                        let all_video_ids: Vec<u64> = seq.video_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
                                        if !all_video_ids.is_empty() {
                                            if let Some(obj) = step.params.as_object_mut() {
                                                obj.insert("clips".to_string(), json!(all_video_ids));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Auto-normalize and validate track and time for timeline.move
                    if step.id == "timeline.move" {
                        if let Some(moves) = step.params.get_mut("moves").and_then(Value::as_array_mut) {
                            if let Some(seq) = session.active_sequence() {
                                for m in moves {
                                    if let Some(m_obj) = m.as_object_mut() {
                                        // 1. Normalize track spec (e.g. "Video 2" -> "V2")
                                        if let Some(t_str) = m_obj.get("track").and_then(Value::as_str) {
                                            let norm_track = super::validation::normalize_track_spec(t_str);
                                            m_obj.insert("track".to_string(), json!(norm_track));
                                        }
                                        // 2. If track is "V<n>" but sequence doesn't have that many video tracks, convert to "new"
                                        if let Some(t_str) = m_obj.get("track").and_then(Value::as_str) {
                                            if let Some(num_str) = t_str.strip_prefix('V') {
                                                if let Ok(n) = num_str.parse::<usize>() {
                                                    if n > seq.video_tracks.len() {
                                                        m_obj.insert("track".to_string(), json!("new"));
                                                    }
                                                }
                                            }
                                        }
                                        // 3. If time is missing or null, default to clip's current start time
                                        let needs_time = m_obj.get("time").map_or(true, |v| v.is_null());
                                        if needs_time {
                                            if let Some(cid) = m_obj.get("clip").and_then(Value::as_u64) {
                                                if let Some((_, it)) = seq.find_item(filmcraft_project::ClipId(cid)) {
                                                    m_obj.insert("time".to_string(), json!(it.start.0));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // 1. Validate all planned steps against engine state
                if let Err(val_err) = validate_steps(session, &plan.steps) {
                    self.error_message = Some(val_err.clone());
                    self.status_message = None;
                    self.history.push(AssistantExchange {
                        user_query: resp.user_query,
                        explanation: plan.explanation,
                        steps: plan
                            .steps
                            .into_iter()
                            .map(|s| StepExecution {
                                id: s.id,
                                params: s.params,
                                success: false,
                                detail: "Validation rejected".to_string(),
                            })
                            .collect(),
                        overall_error: Some(val_err),
                    });
                    return;
                }

                // 2. Automatically execute steps in order
                let mut executed_steps = Vec::new();
                let mut execution_error = None;

                for mut step in plan.steps {
                    let norm_id = super::validation::normalize_command_id(&step.id);

                    // Re-validate clips against live sequence state right before execution
                    if norm_id == "lumetri.applyPreset" {
                        if let Some(seq) = session.active_sequence() {
                            let current_clips: Vec<u64> = step
                                .params
                                .get("clips")
                                .and_then(|c| c.as_array())
                                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
                                .unwrap_or_default();
                            let valid_clips: Vec<u64> = current_clips
                                .into_iter()
                                .filter(|cid| seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id.0 == *cid)))
                                .collect();
                            if valid_clips.is_empty() {
                                let all_video_ids: Vec<u64> = seq
                                    .video_tracks
                                    .iter()
                                    .flat_map(|t| t.items.iter().map(|i| i.id.0))
                                    .collect();
                                if !all_video_ids.is_empty() {
                                    if let Some(obj) = step.params.as_object_mut() {
                                        obj.insert("clips".to_string(), json!(all_video_ids));
                                    }
                                }
                            } else if let Some(obj) = step.params.as_object_mut() {
                                obj.insert("clips".to_string(), json!(valid_clips));
                            }
                        }
                    } else if norm_id == "effects.apply" {
                        if let Some(seq) = session.active_sequence() {
                            let effect_name = step.params.get("effect").and_then(Value::as_str).unwrap_or("");
                            let is_audio = filmcraft_project::find_effect(effect_name)
                                .or_else(|| filmcraft_project::effect_defs().iter().find(|d| d.name.eq_ignore_ascii_case(effect_name)))
                                .is_some_and(|d| d.kind == filmcraft_project::EffectKind::Audio);

                            let current_clips: Vec<u64> = step
                                .params
                                .get("clips")
                                .and_then(|c| c.as_array())
                                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
                                .unwrap_or_default();

                            if is_audio {
                                let valid_clips: Vec<u64> = current_clips
                                    .into_iter()
                                    .filter(|cid| {
                                        if let Some((tid, it)) = seq.find_item(filmcraft_project::ClipId(*cid)) {
                                            let is_audio_track = seq.track(tid).map_or(false, |t| t.kind == filmcraft_project::TrackKind::Audio);
                                            let has_linked_audio = it.link.is_some() && seq.audio_tracks.iter().any(|t| t.items.iter().any(|i| i.link == it.link));
                                            is_audio_track || has_linked_audio
                                        } else {
                                            false
                                        }
                                    })
                                    .collect();
                                if valid_clips.is_empty() {
                                    let all_audio_ids: Vec<u64> = seq.audio_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
                                    if !all_audio_ids.is_empty() {
                                        if let Some(obj) = step.params.as_object_mut() {
                                            obj.insert("clips".to_string(), json!(all_audio_ids));
                                        }
                                    }
                                } else if let Some(obj) = step.params.as_object_mut() {
                                    obj.insert("clips".to_string(), json!(valid_clips));
                                }
                            } else {
                                let valid_clips: Vec<u64> = current_clips
                                    .into_iter()
                                    .filter(|cid| seq.video_tracks.iter().any(|t| t.items.iter().any(|i| i.id.0 == *cid)))
                                    .collect();
                                if valid_clips.is_empty() {
                                    let all_video_ids: Vec<u64> = seq.video_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.id.0)).collect();
                                    if !all_video_ids.is_empty() {
                                        if let Some(obj) = step.params.as_object_mut() {
                                            obj.insert("clips".to_string(), json!(all_video_ids));
                                        }
                                    }
                                } else if let Some(obj) = step.params.as_object_mut() {
                                    obj.insert("clips".to_string(), json!(valid_clips));
                                }
                            }
                        }
                    }

                    match session.execute(&norm_id, step.params.clone()) {
                        Ok(ret_val) => {
                            let detail = if ret_val.is_null() || ret_val == Value::Object(Default::default()) {
                                "Success".to_string()
                            } else {
                                serde_json::to_string(&ret_val).unwrap_or_else(|_| "Success".to_string())
                            };
                            executed_steps.push(StepExecution {
                                id: norm_id,
                                params: step.params,
                                success: true,
                                detail,
                            });
                        }
                        Err(engine_err) => {
                            let err_msg = format!("Engine execution failed on '{}': {}", norm_id, engine_err);
                            executed_steps.push(StepExecution {
                                id: norm_id,
                                params: step.params,
                                success: false,
                                detail: err_msg.clone(),
                            });
                            execution_error = Some(err_msg);
                            break; // Stop immediately upon first failure
                        }
                    }
                }

                if let Some(err) = execution_error {
                    self.error_message = Some(err.clone());
                    self.status_message = None;
                    self.history.push(AssistantExchange {
                        user_query: resp.user_query,
                        explanation: plan.explanation,
                        steps: executed_steps,
                        overall_error: Some(err),
                    });
                } else {
                    let count = executed_steps.len();
                    self.status_message = Some(format!("Successfully executed {count} editing operation(s)."));
                    self.error_message = None;
                    self.history.push(AssistantExchange {
                        user_query: resp.user_query,
                        explanation: plan.explanation,
                        steps: executed_steps,
                        overall_error: None,
                    });
                }
            }
        }
    }

    /// Clear interaction history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.status_message = None;
        self.error_message = None;
    }
}

/// Sanitizes and aligns planned steps with user intent before validation and execution.
pub fn sanitize_plan(user_query: &str, plan: &mut AssistantPlan) {
    let lower = user_query.to_ascii_lowercase();
    let wants_delete = lower.contains("delete") || lower.contains("remove") || lower.contains("cut") || lower.contains("split") || lower.contains("razor") || lower.contains("trim");

    // If user did NOT ask to cut/delete/trim, strip any hallucinated destructive operations
    if !wants_delete {
        plan.steps.retain(|s| {
            let norm = super::validation::normalize_command_id(&s.id);
            norm != "timeline.razor" && norm != "timeline.trim" && norm != "edit.clear" && norm != "edit.rippleDelete"
        });
    }

    // If user did NOT ask to move clips, strip any hallucinated timeline.move
    let wants_move = lower.contains("move")
        || lower.contains("shift")
        || lower.contains("relocate")
        || lower.contains("reposition")
        || lower.contains("track 1")
        || lower.contains("track 2")
        || lower.contains("track 3")
        || lower.split_whitespace().any(|w| {
            let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
            clean == "v1" || clean == "v2" || clean == "v3" || clean == "a1" || clean == "a2" || clean == "a3"
        });
    if !wants_move {
        plan.steps.retain(|s| {
            let norm = super::validation::normalize_command_id(&s.id);
            norm != "timeline.move"
        });
    }

    // If user asked purely for audio processing and did NOT mention video/color, strip hallucinated Lumetri presets
    let is_purely_audio = (lower.contains("audio") || lower.contains("speech") || lower.contains("sound") || lower.contains("voice") || lower.contains("vocal"))
        && !lower.contains("video")
        && !lower.contains("color")
        && !lower.contains("grade")
        && !lower.contains("look")
        && !lower.contains("cinematic");

    if is_purely_audio {
        plan.steps.retain(|s| {
            let norm = super::validation::normalize_command_id(&s.id);
            norm != "lumetri.applyPreset" && norm != "lumetri.setLook" && norm != "lumetri.setInputLut" && norm != "lumetri.setSection"
        });
    }

    // Normalize command IDs and parameters in all steps
    for step in &mut plan.steps {
        step.id = super::validation::normalize_command_id(&step.id);

        if step.id == "lumetri.applyPreset" {
            if let Some(obj) = step.params.as_object_mut() {
                if !obj.contains_key("name") {
                    if let Some(preset_val) = obj.get("preset").cloned() {
                        obj.insert("name".to_string(), preset_val);
                    }
                }
                if let Some(raw_name) = obj.get("name").and_then(Value::as_str) {
                    let norm = super::validation::normalize_preset_name(raw_name);
                    obj.insert("name".to_string(), json!(norm));
                }
            }
        }

        if step.id == "timeline.move" {
            if let Some(moves) = step.params.get_mut("moves").and_then(Value::as_array_mut) {
                for m in moves {
                    if let Some(m_obj) = m.as_object_mut() {
                        if let Some(t_str) = m_obj.get("track").and_then(Value::as_str) {
                            let norm = super::validation::normalize_track_spec(t_str);
                            m_obj.insert("track".to_string(), json!(norm));
                        }
                    }
                }
            }
        }

        if step.id == "effects.apply" {
            if let Some(obj) = step.params.as_object_mut() {
                if !obj.contains_key("effect") {
                    if let Some(name_val) = obj.get("name").cloned() {
                        obj.insert("effect".to_string(), name_val);
                    }
                }
                if let Some(effect_val) = obj.get("effect").and_then(Value::as_str) {
                    let norm = super::validation::normalize_effect_name(effect_val);
                    obj.insert("effect".to_string(), json!(norm));
                }
            }
        }

        // Normalize clip vs clips parameter across lumetri.applyPreset and effects.apply
        if step.id == "lumetri.applyPreset" || step.id == "effects.apply" {
            if let Some(obj) = step.params.as_object_mut() {
                if let Some(c) = obj.get("clip").and_then(Value::as_u64) {
                    if !obj.contains_key("clips") {
                        obj.insert("clips".to_string(), json!([c]));
                    }
                } else if let Some(c) = obj.get("clips").and_then(Value::as_u64) {
                    obj.insert("clips".to_string(), json!([c]));
                }
            }
        }
    }
}
