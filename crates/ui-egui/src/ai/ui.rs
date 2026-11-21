//! Presentation components for the AI Assistant panel in FilmCraft.

use egui::{Color32, CornerRadius, Frame, Margin, Stroke, Vec2};
use crate::FilmcraftApp;

pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    // 1. Poll incoming worker responses and execute validated edits on the UI thread
    let was_processing = app.assistant.is_processing;
    let had_session_dirty = app.session.is_dirty();
    app.assistant.poll_and_execute(&mut app.session);

    // If an edit was executed, trigger repaint and refresh timeline views
    if app.session.is_dirty() && !had_session_dirty {
        ctx.request_repaint();
    }

    // Keep polling while worker is busy
    if app.assistant.is_processing {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    } else if was_processing {
        ctx.request_repaint();
    }

    if !app.assistant.open {
        return;
    }

    let t = app.tokens;
    let mut is_open = app.assistant.open;

    egui::Window::new("AI Editing Assistant")
        .id(egui::Id::new("ai-assistant-window"))
        .open(&mut is_open)
        .default_size(Vec2::new(460.0, 560.0))
        .min_width(340.0)
        .min_height(380.0)
        .resizable(true)
        .collapsible(true)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);

            // Header status strip
            ui.horizontal(|ui| {
                let dot_color = if app.assistant.is_processing {
                    Color32::from_rgb(255, 193, 7) // Yellow
                } else if app.assistant.error_message.is_some() {
                    Color32::from_rgb(244, 67, 54) // Red
                } else {
                    Color32::from_rgb(76, 175, 80) // Green
                };
                let (rect, _) = ui.allocate_exact_size(Vec2::new(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.5, dot_color);

                ui.label(
                    egui::RichText::new(format!("Local Model: {}", app.assistant.model))
                        .size(11.0)
                        .color(t.text_dim),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").on_hover_text("Clear history").clicked() {
                        app.assistant.clear_history();
                    }
                });
            });

            ui.separator();

            // History / Conversation scroll area
            let available_height = ui.available_height() - 140.0;
            egui::ScrollArea::vertical()
                .max_height(available_height.max(160.0))
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if app.assistant.history.is_empty() {
                        ui.add_space(20.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new("FilmCraft AI Editing Assistant")
                                    .size(14.0)
                                    .strong()
                                    .color(t.text),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Connected to local Ollama (qwen2.5:3b).\nEnter natural language commands to edit the timeline.",
                                )
                                .size(12.0)
                                .color(t.text_dim),
                            );
                        });
                        ui.add_space(20.0);
                    } else {
                        for exchange in &app.assistant.history {
                            // User prompt bubble
                            Frame::NONE
                                .fill(t.hover)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::same(8))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("User:").strong().size(12.0).color(t.accent));
                                        ui.label(egui::RichText::new(&exchange.user_query).size(12.0).color(t.text));
                                    });
                                });

                            ui.add_space(4.0);

                            // Assistant response bubble
                            Frame::NONE
                                .fill(t.header_bg)
                                .stroke(Stroke::new(1.0, t.separator))
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::same(8))
                                .show(ui, |ui| {
                                    if !exchange.explanation.is_empty() {
                                        ui.label(
                                            egui::RichText::new(&exchange.explanation)
                                                .size(12.0)
                                                .color(t.text),
                                        );
                                        ui.add_space(4.0);
                                    }

                                    // Executed steps
                                    if !exchange.steps.is_empty() {
                                        ui.label(
                                            egui::RichText::new("Executed Engine Operations:")
                                                .size(11.0)
                                                .strong()
                                                .color(t.text_dim),
                                        );
                                        for s in &exchange.steps {
                                            ui.horizontal(|ui| {
                                                if s.success {
                                                    ui.label(
                                                        egui::RichText::new("✓")
                                                            .color(Color32::from_rgb(76, 175, 80))
                                                            .strong(),
                                                    );
                                                } else {
                                                    ui.label(
                                                        egui::RichText::new("✗")
                                                            .color(Color32::from_rgb(244, 67, 54))
                                                            .strong(),
                                                    );
                                                }
                                                ui.label(
                                                    egui::RichText::new(&s.id)
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(if s.success { t.text } else { Color32::from_rgb(244, 67, 54) }),
                                                );
                                                if !s.detail.is_empty() && s.detail != "Success" {
                                                    ui.label(
                                                        egui::RichText::new(format!("({})", s.detail))
                                                            .size(10.0)
                                                            .color(t.text_dim),
                                                    );
                                                }
                                            });
                                        }
                                    }

                                    // Error display
                                    if let Some(err) = &exchange.overall_error {
                                        ui.add_space(4.0);
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new(format!("⚠ Error: {err}"))
                                                    .size(11.0)
                                                    .color(Color32::from_rgb(244, 67, 54)),
                                            );
                                        });
                                    }
                                });

                            ui.add_space(8.0);
                        }
                    }
                });

            ui.separator();

            // Quick suggestion chips
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Quick:").size(10.0).color(t.text_dim));

                let presets = [
                    ("✂ Cut first 5s", "Cut the first 5 seconds of the selected clip."),
                    ("📝 Title: Welcome", "Add a title saying \"Welcome\"."),
                    ("🎨 Color grading", "Apply color grading to the video."),
                    ("🎞 Cinematic", "Make the video more cinematic."),
                    ("⏮ Undo", "Undo the last edit."),
                ];

                for (i, (label, prompt_text)) in presets.into_iter().enumerate() {
                    let btn = ui.add_enabled(!app.assistant.is_processing, egui::Button::new(egui::RichText::new(label).size(10.0)));
                    app.auto.add_response(&format!("ai.preset.{i}"), &btn, label);
                    if btn.clicked() {
                        app.assistant.submit_preset(&app.session, prompt_text);
                    }
                }
            });

            // Status / Error banner
            if app.assistant.is_processing {
                ui.horizontal(|ui| {
                    ui.spinner();
                    let msg = app.assistant.status_message.as_deref().unwrap_or("Thinking & generating edits...");
                    ui.label(egui::RichText::new(msg).size(11.0).color(t.accent));
                });
            } else if let Some(err) = &app.assistant.error_message {
                ui.label(
                    egui::RichText::new(format!("⚠ {err}"))
                        .size(11.0)
                        .color(Color32::from_rgb(244, 67, 54)),
                );
            } else if let Some(status) = &app.assistant.status_message {
                ui.label(
                    egui::RichText::new(status)
                        .size(11.0)
                        .color(Color32::from_rgb(76, 175, 80)),
                );
            }

            // Input area
            ui.horizontal(|ui| {
                let text_edit = egui::TextEdit::multiline(&mut app.assistant.prompt_input)
                    .desired_rows(2)
                    .desired_width(ui.available_width() - 75.0)
                    .hint_text("Type an editing instruction (e.g. 'Cut first 5 seconds', 'Add title Welcome')...");

                let response = ui.add_enabled(!app.assistant.is_processing, text_edit);
                app.auto.add_response("ai.input", &response, "AI Instruction Input");

                // Enter key sends the prompt (unless shift is pressed)
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift) {
                    app.assistant.submit_prompt(&app.session);
                }

                let send_btn = ui.add_enabled(
                    !app.assistant.is_processing && !app.assistant.prompt_input.trim().is_empty(),
                    egui::Button::new(egui::RichText::new("Execute").strong())
                        .min_size(Vec2::new(65.0, 34.0)),
                );
                app.auto.add_response("ai.execute", &send_btn, "Execute");

                if send_btn.clicked() {
                    app.assistant.submit_prompt(&app.session);
                }
            });
        });

    app.assistant.open = is_open;
}
