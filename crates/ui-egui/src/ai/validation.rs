//! Validates AI-generated commands against actual FilmCraft engine state and schemas.

use filmcraft_engine::Session;
use filmcraft_engine::commands::command_specs;
use filmcraft_project::{ClipId, TrackId};
use serde_json::Value;

use super::client::PlannedStep;

/// Whitelist of safe editing commands the AI Assistant is allowed to execute.
pub(crate) const ALLOWED_COMMANDS: &[&str] = &[
    "timeline.razor",
    "timeline.trim",
    "timeline.move",
    "timeline.select",
    "timeline.setTrack",
    "edit.clear",
    "edit.rippleDelete",
    "graphics.newText",
    "graphics.setText",
    "graphics.set",
    "graphics.newShape",
    "lumetri.applyPreset",
    "lumetri.setLook",
    "lumetri.setInputLut",
    "lumetri.setSection",
    "effects.apply",
    "effects.remove",
    "effects.setParam",
    "clip.speedDuration",
    "playhead.set",
    "edit.undo",
    "edit.redo",
    "file.save",
    "file.openDemoProject",
    "file.newSequence",
    "sequence.settings",
    "project.inspect",
    "sequence.inspect",
    "command.list",
];

/// Normalizes command IDs in case the LLM appends step suffixes or uses common aliases.
pub fn normalize_command_id(raw_id: &str) -> String {
    let trimmed = raw_id.trim();
    if ALLOWED_COMMANDS.contains(&trimmed) {
        return trimmed.to_string();
    }
    // Aliases to actual registered engine commands
    match trimmed {
        "timeline.rippleDelete" | "timeline.ripple_delete" | "rippleDelete" => return "edit.rippleDelete".to_string(),
        "timeline.delete" | "timeline.clear" | "edit.delete" | "delete" => return "edit.clear".to_string(),
        "playback.seek" | "timeline.seek" | "seek" => return "playhead.set".to_string(),
        _ => {}
    }
    // Check if stripping trailing suffix matches an allowed command
    if let Some((base, _)) = trimmed.split_once('-') {
        let norm_base = normalize_command_id(base);
        if ALLOWED_COMMANDS.contains(&norm_base.as_str()) {
            return norm_base;
        }
    }
    if let Some((base, _)) = trimmed.split_once('_') {
        let norm_base = normalize_command_id(base);
        if ALLOWED_COMMANDS.contains(&norm_base.as_str()) {
            return norm_base;
        }
    }
    trimmed.to_string()
}

/// Normalizes Lumetri preset names to actual registered presets in `filmcraft_render::lumetri_presets`.
pub fn normalize_preset_name(raw_name: &str) -> String {
    let trimmed = raw_name.trim();
    if filmcraft_render::lumetri_presets::find(trimmed).is_some() {
        return trimmed.to_string();
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "cinematic" | "cinema" | "teal and orange" | "teal & orange" | "default" => {
            "Teal Shadows, Warm Skin".to_string()
        }
        "night exterior" | "night" | "day for night" => "Night Exterior".to_string(),
        "golden dusk" | "dusk" | "golden hour" | "sunset" | "warm dusk" => "Golden Dusk".to_string(),
        "steel blue" | "steel" | "cool blue" | "cold blue" => "Steel Blue".to_string(),
        "hard sun" | "sun" | "sunlight" => "Hard Sun".to_string(),
        "faded print" | "faded film" | "vintage" | "fade" => "Faded Print".to_string(),
        "silver retention" | "silver" | "bleach bypass" => "Silver Retention".to_string(),
        "cross processed" | "cross process" => "Cross Processed".to_string(),
        "warm negative" | "negative" | "film emulation" | "film" => "Warm Negative".to_string(),
        "neutral mono" | "monochrome" | "mono" | "black and white" | "b&w" => "Neutral Mono".to_string(),
        "hard mono" | "contrast mono" => "Hard Mono".to_string(),
        "sepia tone" | "sepia" => "Sepia Tone".to_string(),
        "cold mono" | "cyan mono" => "Cold Mono".to_string(),
        "boost vibrance" | "vibrance" | "technical" => "Boost Vibrance".to_string(),
        "half saturation" | "desaturate" => "Half Saturation".to_string(),
        "lift shadows" => "Lift Shadows".to_string(),
        "protect highlights" => "Protect Highlights".to_string(),
        "legal range squeeze" => "Legal Range Squeeze".to_string(),
        _ => trimmed.to_string(),
    }
}

/// Normalizes track specifications (e.g. "Video 2", "Video 1", "Track 2", "track 1", "Audio 1")
/// into engine-accepted track formats ("V2", "V1", "A1", or "new").
pub fn normalize_track_spec(raw: &str) -> String {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();

    if lower == "new" {
        return "new".to_string();
    }

    // Handle "video 2", "video2", "v2"
    if let Some(rest) = lower.strip_prefix("video") {
        let num_str = rest.trim().trim_start_matches(|c: char| !c.is_ascii_digit());
        if let Ok(n) = num_str.parse::<usize>() {
            return format!("V{n}");
        }
    }

    // Handle "audio 2", "audio2", "a2"
    if let Some(rest) = lower.strip_prefix("audio") {
        let num_str = rest.trim().trim_start_matches(|c: char| !c.is_ascii_digit());
        if let Ok(n) = num_str.parse::<usize>() {
            return format!("A{n}");
        }
    }

    // Handle "track 2", "track2"
    if let Some(rest) = lower.strip_prefix("track") {
        let num_str = rest.trim().trim_start_matches(|c: char| !c.is_ascii_digit());
        if let Ok(n) = num_str.parse::<usize>() {
            return format!("V{n}");
        }
    }

    // Already in V<n> or A<n> format
    if (trimmed.starts_with('V') || trimmed.starts_with('v') || trimmed.starts_with('A') || trimmed.starts_with('a'))
        && trimmed.len() > 1
        && trimmed[1..].trim().parse::<usize>().is_ok()
    {
        let prefix = trimmed.chars().next().unwrap_or('V').to_ascii_uppercase();
        let num = trimmed[1..].trim();
        return format!("{prefix}{num}");
    }

    trimmed.to_string()
}

/// Normalizes effect names and common aliases to actual registered effect IDs in `filmcraft_project`.
pub fn normalize_effect_name(raw: &str) -> String {
    let trimmed = raw.trim();
    if filmcraft_project::find_effect(trimmed).is_some() {
        return trimmed.to_string();
    }
    let lower_spaced = trimmed.to_ascii_lowercase().replace(['-', '_'], " ");
    match lower_spaced.as_str() {
        // Audio effect aliases
        "bass boost" | "bassboost" | "bass" | "sub bass" | "subbass" | "audio impact" | "impact" => "bass".to_string(),
        "treble boost" | "trebleboost" | "treble" => "treble".to_string(),
        "compressor" | "tube compressor" | "audio compressor" => "tube_compressor".to_string(),
        "limiter" | "hard limiter" => "hard_limiter".to_string(),
        "reverb" | "studio reverb" | "room reverb" => "studio_reverb".to_string(),
        "echo" | "delay" | "analog delay" => "delay".to_string(),
        "denoise" | "noise reduction" | "de noise" | "clean noise" => "denoise".to_string(),
        "dereverb" | "echo reduction" | "de reverb" | "remove echo" => "dereverb".to_string(),
        "vocal enhancer" | "vocal boost" | "voice enhancer" | "speech clarity" => "vocal_enhancer".to_string(),
        "parametric eq" | "eq" | "equalizer" => "parametric_eq".to_string(),
        "pitch shifter" | "pitch shift" => "pitch_shifter".to_string(),

        // Video effect aliases
        "camera shake" | "camera shaking" | "shake" | "screen shake" | "handheld" => "camera_shake".to_string(),
        "strobe" | "strobe light" | "strobe_light" | "lighting on off" | "light on off" | "flash" | "flashing" | "flicker" => "strobe".to_string(),
        "rgb split" | "chromatic aberration" | "glitch" | "color split" => "rgb_split".to_string(),
        "gaussian blur" | "blur" | "defocus" => "gaussian_blur".to_string(),
        "vignette" | "lens vignette" => "vignette".to_string(),
        "lens flare" | "flare" => "lens_flare".to_string(),
        "black and white" | "black & white" | "black white" | "grayscale" | "b&w" => "black_white".to_string(),
        "crop" | "letterbox" => "crop".to_string(),
        "sharpen" => "sharpen".to_string(),
        "mosaic" | "pixelate" | "pixelation" => "mosaic".to_string(),
        _ => trimmed.to_string(),
    }
}

/// Validates a list of steps generated by the AI against engine schemas and project state.
pub fn validate_steps(session: &Session, steps: &[PlannedStep]) -> Result<(), String> {
    if steps.is_empty() {
        return Err("The assistant returned an empty plan with no editing steps.".to_string());
    }

    let all_specs = command_specs();

    for (index, step) in steps.iter().enumerate() {
        let step_num = index + 1;
        let norm_id = normalize_command_id(&step.id);

        // 1. Whitelist check
        if !ALLOWED_COMMANDS.contains(&norm_id.as_str()) {
            return Err(format!(
                "Step {step_num}: Command '{}' is not in the allowed editing commands whitelist.",
                step.id
            ));
        }

        // 2. Existence check in engine specs
        let spec = all_specs
            .iter()
            .find(|s| s.id == norm_id.as_str())
            .ok_or_else(|| format!("Step {step_num}: Command '{}' is not registered in the engine.", norm_id))?;

        // 3. Command enablement check
        if let Err(reason) = (spec.enabled)(session) {
            return Err(format!(
                "Step {step_num}: Command '{}' cannot be executed in current state: {reason}",
                norm_id
            ));
        }

        // 4. Parameter sanity checks
        validate_step_params(session, step_num, &step.id, &step.params)?;
    }

    Ok(())
}

/// Validates parameters for a specific command step.
fn validate_step_params(session: &Session, step_num: usize, command_id: &str, params: &Value) -> Result<(), String> {
    // If command requires active sequence, check clip references
    if let Some(seq_id) = session.state.active_sequence {
        if let Some(seq) = session.project.sequence(seq_id) {
            // Check single "clip": <id>
            if let Some(clip_val) = params.get("clip") {
                if let Some(cid) = clip_val.as_u64() {
                    let clip_id = ClipId(cid);
                    if seq.find_item(clip_id).is_none() {
                        return Err(format!(
                            "Step {step_num} ({command_id}): Referenced clip ID {cid} does not exist in the active sequence."
                        ));
                    }
                }
            }

            // Check array "clips": [<id>, ...]
            if let Some(clips_val) = params.get("clips").and_then(Value::as_array) {
                for c in clips_val {
                    if let Some(cid) = c.as_u64() {
                        let clip_id = ClipId(cid);
                        if seq.find_item(clip_id).is_none() {
                            return Err(format!(
                                "Step {step_num} ({command_id}): Referenced clip ID {cid} does not exist in the active sequence."
                            ));
                        }
                    }
                }
            }

            // Check "moves": [{"clip": <id>, "track": <t>, ...}]
            if let Some(moves) = params.get("moves").and_then(Value::as_array) {
                for m in moves {
                    if let Some(cid) = m.get("clip").and_then(Value::as_u64) {
                        let clip_id = ClipId(cid);
                        if seq.find_item(clip_id).is_none() {
                            return Err(format!(
                                "Step {step_num} ({command_id}): Move target clip ID {cid} does not exist in sequence."
                            ));
                        }
                    }
                }
            }

            // Check "track": <id> if numeric
            if let Some(track_val) = params.get("track").and_then(Value::as_u64) {
                let tid = TrackId(track_val);
                let exists = seq.video_tracks.iter().chain(seq.audio_tracks.iter()).any(|t| t.id == tid);
                if !exists {
                    return Err(format!(
                        "Step {step_num} ({command_id}): Track ID {track_val} does not exist in the active sequence."
                    ));
                }
            }
        }
    }

    // Check Lumetri preset existence
    if command_id == "lumetri.applyPreset" {
        let name_str = params.get("name").or_else(|| params.get("preset")).and_then(Value::as_str);
        if let Some(name_val) = name_str {
            let norm_name = normalize_preset_name(name_val);
            if filmcraft_render::lumetri_presets::find(&norm_name).is_none() {
                return Err(format!(
                    "Step {step_num} ({command_id}): No Lumetri preset '{name_val}'. Valid presets include: 'Teal Shadows, Warm Skin', 'Golden Dusk', 'Night Exterior', 'Steel Blue', 'Faded Print', 'Neutral Mono'."
                ));
            }
        }
    }

    Ok(())
}
