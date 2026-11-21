//! Local AI Editing Assistant for FilmCraft.
//!
//! Provides a natural language interface backed by local Ollama (`qwen2.5:3b`), translating
//! user instructions into validated engine editing commands executed on the timeline.

pub mod client;
pub mod prompt;
pub mod state;
pub mod ui;
pub mod validation;

pub use client::{AssistantPlan, OllamaClient, PlannedStep};
pub use state::AssistantState;

#[cfg(test)]
mod tests;
