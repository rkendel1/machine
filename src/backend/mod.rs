//! Inference backends.
//!
//! Every backend returns the same [`Generation`] record so experiments can be
//! compared across backends without hiding where the time went. Timings that
//! come from the server (prefill / decode) are kept separate from the
//! client-measured wall clock, so HTTP + JSON overhead stays visible.

pub mod llama_server;
pub mod mock;
pub mod ollama;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Per-request generation parameters. Only things that can change *per call*
/// live here. Load-time settings (quantization, GPU offload layers, threads,
/// context size) belong to the server process and are recorded as tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenParams {
    pub max_tokens: u32,
    /// 0.0 = greedy decode.
    pub temperature: f32,
    pub seed: Option<u64>,
    /// llama-server only: reuse the KV cache for a shared prompt prefix.
    pub cache_prompt: bool,
    /// Stop strings (e.g. "\n" for one-line agent replies).
    pub stop: Vec<String>,
    /// llama-server only: keep generating past end-of-sequence so every run
    /// produces exactly max_tokens (needed for clean max_tokens sweeps).
    #[serde(default)]
    pub ignore_eos: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            seed: Some(42),
            cache_prompt: false,
            stop: Vec::new(),
            ignore_eos: false,
        }
    }
}

/// What one inference call produced and what it cost.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Generation {
    pub text: String,
    /// Prompt tokens actually evaluated (cached tokens are not counted by most servers).
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Server-reported prefill time (prompt processing), ms.
    pub prefill_ms: Option<f64>,
    /// Server-reported decode time (token generation), ms.
    pub decode_ms: Option<f64>,
    /// Server-reported model load time, ms (Ollama reports this; llama-server does not).
    pub load_ms: Option<f64>,
    /// Client-measured wall clock for the whole request, ms.
    pub wall_ms: f64,
    /// Sampled peak RSS of the watched process during the call, bytes.
    pub peak_rss_bytes: Option<u64>,
    /// RSS of the watched process right before the call, bytes.
    pub start_rss_bytes: Option<u64>,
    /// True if numbers come from a simulator rather than real hardware.
    pub simulated: bool,
}

impl Generation {
    /// Decode throughput: completion tokens per second of decode time.
    /// Falls back to wall clock when the server does not report decode time.
    pub fn decode_tok_s(&self) -> f64 {
        let ms = self.decode_ms.unwrap_or(self.wall_ms);
        if ms <= 0.0 {
            0.0
        } else {
            self.completion_tokens as f64 * 1000.0 / ms
        }
    }

    /// Prefill throughput: prompt tokens per second of prefill time.
    pub fn prefill_tok_s(&self) -> Option<f64> {
        let ms = self.prefill_ms?;
        if ms <= 0.0 || self.prompt_tokens == 0 {
            None
        } else {
            Some(self.prompt_tokens as f64 * 1000.0 / ms)
        }
    }

    /// Wall time not explained by prefill + decode (HTTP, JSON, scheduling, load).
    pub fn overhead_ms(&self) -> Option<f64> {
        Some(self.wall_ms - self.prefill_ms? - self.decode_ms?)
    }
}

pub trait Backend {
    fn name(&self) -> String;
    fn generate(&mut self, prompt: &str, params: &GenParams) -> Result<Generation>;
    /// Short description of the loaded model, if the backend can report it.
    fn model_info(&mut self) -> Option<String> {
        None
    }
    /// Process name to watch for memory, if the model lives in another process.
    fn server_process_hint(&self) -> Option<&'static str> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum BackendKind {
    /// llama.cpp `llama-server` HTTP API (/completion).
    LlamaServer,
    /// Ollama HTTP API (/api/generate).
    Ollama,
    /// Deterministic simulator. Numbers are NOT hardware measurements.
    Mock,
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub kind: BackendKind,
    pub url: Option<String>,
    pub model: Option<String>,
}

pub fn build(cfg: &BackendConfig) -> Result<Box<dyn Backend>> {
    Ok(match cfg.kind {
        BackendKind::LlamaServer => Box::new(llama_server::LlamaServer::new(
            cfg.url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8080".into()),
        )),
        BackendKind::Ollama => {
            let model = cfg.model.clone().ok_or_else(|| {
                anyhow::anyhow!("--model is required for ollama (e.g. --model qwen2.5:0.5b)")
            })?;
            Box::new(ollama::Ollama::new(
                cfg.url
                    .clone()
                    .unwrap_or_else(|| "http://127.0.0.1:11434".into()),
                model,
            ))
        }
        BackendKind::Mock => Box::new(mock::Mock::default()),
    })
}

pub(crate) fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(600))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_math() {
        let g = Generation {
            prompt_tokens: 100,
            completion_tokens: 50,
            prefill_ms: Some(200.0),
            decode_ms: Some(1000.0),
            wall_ms: 1250.0,
            ..Default::default()
        };
        assert!((g.decode_tok_s() - 50.0).abs() < 1e-9);
        assert!((g.prefill_tok_s().unwrap() - 500.0).abs() < 1e-9);
        assert!((g.overhead_ms().unwrap() - 50.0).abs() < 1e-9);
    }
}
