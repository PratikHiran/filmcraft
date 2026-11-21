//! HTTP communication with the local Ollama daemon for the AI Assistant.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An editing step planned by the AI model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedStep {
    pub id: String,
    #[serde(default)]
    pub params: Value,
}

/// The structured response returned by the local LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantPlan {
    #[serde(default)]
    pub explanation: String,
    #[serde(default)]
    pub steps: Vec<PlannedStep>,
}

/// Client configuration for Ollama.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub endpoint: String,
    pub model: String,
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".to_string(),
            model: "qwen2.5:3b".to_string(),
        }
    }
}

impl OllamaClient {
    pub fn new(endpoint: String, model: String) -> Self {
        Self { endpoint, model }
    }

    /// Parse host and port from endpoint URL.
    fn parse_addr(&self) -> (String, u16) {
        let trimmed = self.endpoint.trim();
        let without_proto = if let Some(stripped) = trimmed.strip_prefix("http://") {
            stripped
        } else if let Some(stripped) = trimmed.strip_prefix("https://") {
            stripped
        } else {
            trimmed
        };
        let host_port = without_proto.trim_end_matches('/');
        if let Some((h, p)) = host_port.split_once(':') {
            let port = p.parse::<u16>().unwrap_or(11434);
            (h.to_string(), port)
        } else {
            (host_port.to_string(), 11434)
        }
    }

    /// Connect to Ollama, trying the configured host first, then 127.0.0.1 fallback.
    fn connect(&self, timeout: Duration) -> Result<TcpStream, String> {
        let (host, port) = self.parse_addr();
        let conn_timeout = Duration::from_secs(3);

        // Try resolved addresses first
        if let Ok(iter) = (host.as_str(), port).to_socket_addrs() {
            for addr in iter {
                if let Ok(stream) = TcpStream::connect_timeout(&addr, conn_timeout) {
                    let _ = stream.set_read_timeout(Some(timeout));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
                    return Ok(stream);
                }
            }
        }

        // Fallback to explicit IPv4 loopback
        let loopback = SocketAddr::from(([127, 0, 0, 1], port));
        if let Ok(stream) = TcpStream::connect_timeout(&loopback, conn_timeout) {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
            return Ok(stream);
        }

        Err(format!("Could not connect to Ollama at {host}:{port} within timeout"))
    }

    /// Check if Ollama is reachable and returns tags.
    pub fn check_health(&self) -> Result<(), String> {
        let mut stream = self.connect(Duration::from_secs(5))?;
        let (host, port) = self.parse_addr();
        let req = format!(
            "GET /api/tags HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        );
        stream
            .write_all(req.as_bytes())
            .map_err(|e| format!("Failed to send health check to Ollama: {e}"))?;

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read health check response: {e}"))?;

        let text = String::from_utf8_lossy(&buf);
        if text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200") {
            Ok(())
        } else {
            Err(format!("Ollama health check returned non-200 response: {text}"))
        }
    }

    /// Send a prompt to Ollama's /api/generate endpoint with JSON schema mode enabled.
    pub fn generate(&self, system: &str, prompt: &str) -> Result<AssistantPlan, String> {
        let mut stream = self.connect(Duration::from_secs(120))?;
        let (host, port) = self.parse_addr();

        // Enforce strict schema with Ollama's grammar-guided generation
        let plan_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "explanation": { "type": "string" },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": [
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
                                    "file.save"
                                ]
                            },
                            "params": { "type": "object" }
                        },
                        "required": ["id", "params"]
                    }
                }
            },
            "required": ["explanation", "steps"]
        });

        let request_body = serde_json::json!({
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": false,
            "format": plan_schema,
            "options": {
                "temperature": 0.1
            }
        });

        let body_str = serde_json::to_string(&request_body)
            .map_err(|e| format!("Failed to serialize request body: {e}"))?;

        let req = format!(
            "POST /api/generate HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body_str.len(),
            body_str
        );

        stream
            .write_all(req.as_bytes())
            .map_err(|e| format!("Failed to send request to Ollama: {e}"))?;

        let mut response_bytes = Vec::new();
        stream
            .read_to_end(&mut response_bytes)
            .map_err(|e| format!("Failed to read response from Ollama: {e}"))?;

        let response_str = String::from_utf8_lossy(&response_bytes);

        // Split HTTP headers and body
        let raw_body = if let Some((_, b)) = response_str.split_once("\r\n\r\n") {
            b
        } else if let Some((_, b)) = response_str.split_once("\n\n") {
            b
        } else {
            return Err("Malformed HTTP response from Ollama".to_string());
        };

        // Decode chunked transfer encoding if present
        let decoded_body = decode_chunked_body(raw_body);
        let clean_outer_json = extract_json_object(&decoded_body);

        // Parse outer Ollama response object
        let ollama_resp: Value = serde_json::from_str(clean_outer_json)
            .map_err(|e| format!("Failed to parse Ollama JSON: {e} (body: {clean_outer_json})"))?;

        let inner_response_str = ollama_resp
            .get("response")
            .and_then(Value::as_str)
            .ok_or_else(|| "Ollama response missing 'response' field".to_string())?;

        // Clean markdown backticks and locate JSON object if any
        let clean_json = extract_json_object(strip_code_fences(inner_response_str));

        // Parse structured AssistantPlan
        let plan: AssistantPlan = serde_json::from_str(clean_json)
            .map_err(|e| format!("Failed to parse assistant plan JSON: {e} (raw text: {clean_json})"))?;

        Ok(plan)
    }
}

/// Decode HTTP chunked transfer encoding if present, or return body as-is.
pub fn decode_chunked_body(raw_body: &str) -> String {
    let trimmed = raw_body.trim();
    if let Some((first_line, _)) = trimmed.split_once("\r\n") {
        let chunk_hex = first_line.trim().split(';').next().unwrap_or("").trim();
        if !chunk_hex.is_empty() && chunk_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            if let Ok(first_chunk_size) = usize::from_str_radix(chunk_hex, 16) {
                if first_chunk_size > 0 {
                    let mut result = String::new();
                    let mut remaining = trimmed;
                    while !remaining.is_empty() {
                        if let Some((size_line, data_rest)) = remaining.split_once("\r\n") {
                            let hex = size_line.trim().split(';').next().unwrap_or("").trim();
                            if let Ok(size) = usize::from_str_radix(hex, 16) {
                                if size == 0 {
                                    break;
                                }
                                if data_rest.len() >= size {
                                    result.push_str(&data_rest[..size]);
                                    let after = &data_rest[size..];
                                    remaining = after.strip_prefix("\r\n").unwrap_or(after);
                                } else {
                                    result.push_str(data_rest);
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if !result.is_empty() {
                        return result;
                    }
                }
            }
        }
    }
    raw_body.to_string()
}

/// Extract outermost JSON object from a string, handling any extraneous leading/trailing text.
pub fn extract_json_object(text: &str) -> &str {
    let trimmed = text.trim();
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            return &trimmed[start..=end];
        }
    }
    trimmed
}

/// Strip markdown fences like ```json ... ``` from the LLM output.
pub fn strip_code_fences(text: &str) -> &str {
    let mut s = text.trim();
    if s.starts_with("```json") {
        s = s.strip_prefix("```json").unwrap_or(s);
    } else if s.starts_with("```") {
        s = s.strip_prefix("```").unwrap_or(s);
    }
    if s.ends_with("```") {
        s = s.strip_suffix("```").unwrap_or(s);
    }
    s.trim()
}
