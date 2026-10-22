//! llama.cpp `llama-server` backend.
//!
//! Start the server yourself, e.g.:
//!   llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 4096 -t 8 --port 8080
//! Load-time knobs (-m quant, -ngl, -t, -c) are fixed for the server's
//! lifetime. Change one, restart, and tag the run (`--tag quant=q8_0`).

use super::{Backend, GenParams, Generation, http_agent};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::Instant;

pub struct LlamaServer {
    base: String,
    agent: ureq::Agent,
}

impl LlamaServer {
    pub fn new(base: String) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            agent: http_agent(),
        }
    }
}

fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

impl Backend for LlamaServer {
    fn name(&self) -> String {
        format!("llama-server@{}", self.base)
    }

    fn generate(&mut self, prompt: &str, p: &GenParams) -> Result<Generation> {
        let mut body = json!({
            "prompt": prompt,
            "n_predict": p.max_tokens,
            "temperature": p.temperature,
            "cache_prompt": p.cache_prompt,
            "stream": false,
        });
        if let Some(seed) = p.seed {
            body["seed"] = json!(seed);
        }
        if p.ignore_eos {
            body["ignore_eos"] = json!(true);
        }
        if !p.stop.is_empty() {
            body["stop"] = json!(p.stop);
        }
        let t0 = Instant::now();
        let resp: Value = self
            .agent
            .post(&format!("{}/completion", self.base))
            .send_json(body)
            .with_context(|| format!("POST {}/completion (is llama-server running?)", self.base))?
            .into_json()?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let timings = resp.get("timings").cloned().unwrap_or(Value::Null);
        let completion_tokens = f(&timings, "predicted_n")
            .or_else(|| f(&resp, "tokens_predicted"))
            .unwrap_or(0.0) as u64;
        let prompt_tokens = f(&timings, "prompt_n")
            .or_else(|| f(&resp, "tokens_evaluated"))
            .unwrap_or(0.0) as u64;

        Ok(Generation {
            text: resp
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            prompt_tokens,
            completion_tokens,
            prefill_ms: f(&timings, "prompt_ms"),
            decode_ms: f(&timings, "predicted_ms"),
            load_ms: None,
            wall_ms,
            simulated: false,
            ..Default::default()
        })
    }

    fn model_info(&mut self) -> Option<String> {
        let v: Value = self
            .agent
            .get(&format!("{}/props", self.base))
            .call()
            .ok()?
            .into_json()
            .ok()?;
        let path = v.get("model_path").and_then(Value::as_str).unwrap_or("?");
        let n_ctx = v
            .pointer("/default_generation_settings/n_ctx")
            .and_then(Value::as_u64)
            .map(|n| format!(" n_ctx={n}"))
            .unwrap_or_default();
        Some(format!("{path}{n_ctx}"))
    }

    fn server_process_hint(&self) -> Option<&'static str> {
        Some("llama-server")
    }
}
