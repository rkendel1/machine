//! Ollama backend (/api/generate, non-streaming).
//!
//! Ollama reports durations in nanoseconds: load_duration, prompt_eval_duration,
//! eval_duration, plus prompt_eval_count and eval_count. If the model was not
//! resident, the first call includes load time — watch `load_ms`.

use super::{Backend, GenParams, Generation, http_agent};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::Instant;

pub struct Ollama {
    base: String,
    model: String,
    agent: ureq::Agent,
}

impl Ollama {
    pub fn new(base: String, model: String) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            model,
            agent: http_agent(),
        }
    }
}

fn ns_to_ms(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64).map(|ns| ns / 1e6)
}

impl Backend for Ollama {
    fn name(&self) -> String {
        format!("ollama:{}@{}", self.model, self.base)
    }

    fn generate(&mut self, prompt: &str, p: &GenParams) -> Result<Generation> {
        let mut options = json!({
            "temperature": p.temperature,
            "num_predict": p.max_tokens,
        });
        if let Some(seed) = p.seed {
            options["seed"] = json!(seed);
        }
        if !p.stop.is_empty() {
            options["stop"] = json!(p.stop);
        }
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            // raw: no chat template, so the prompt you see is the prompt the model sees.
            "raw": true,
            "options": options,
        });
        let t0 = Instant::now();
        let resp: Value = self
            .agent
            .post(&format!("{}/api/generate", self.base))
            .send_json(body)
            .with_context(|| format!("POST {}/api/generate (is ollama running?)", self.base))?
            .into_json()?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(Generation {
            text: resp
                .get("response")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            prompt_tokens: resp
                .get("prompt_eval_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: resp.get("eval_count").and_then(Value::as_u64).unwrap_or(0),
            prefill_ms: ns_to_ms(&resp, "prompt_eval_duration"),
            decode_ms: ns_to_ms(&resp, "eval_duration"),
            load_ms: ns_to_ms(&resp, "load_duration"),
            wall_ms,
            simulated: false,
            ..Default::default()
        })
    }

    fn model_info(&mut self) -> Option<String> {
        let v: Value = self
            .agent
            .post(&format!("{}/api/show", self.base))
            .send_json(json!({ "model": self.model }))
            .ok()?
            .into_json()
            .ok()?;
        let d = v.get("details")?;
        let get = |k: &str| d.get(k).and_then(Value::as_str).unwrap_or("?");
        Some(format!(
            "{} params={} quant={}",
            self.model,
            get("parameter_size"),
            get("quantization_level")
        ))
    }

    fn server_process_hint(&self) -> Option<&'static str> {
        Some("ollama")
    }
}
