//! Deterministic simulator backend.
//!
//! Use it to learn the harness, write tests, and rehearse experiments when no
//! model is available. Every number it produces is SIMULATED from a simple
//! cost model — it is never evidence about your hardware. Results carry
//! `simulated: true` and the CLI prints a banner.
//!
//! Cost model (per call):
//!   prefill_ms = prompt_tokens * prefill_ms_per_tok
//!   decode_ms  = completion_tokens * (decode_ms_per_tok + ctx_penalty * context_len)
//!   both multiplied by (1 ± jitter) and, for sampling, a small overhead.

use super::{Backend, GenParams, Generation};
use crate::agent;
use anyhow::Result;
use std::time::{Duration, Instant};

pub struct Mock {
    pub prefill_ms_per_tok: f64,
    pub decode_ms_per_tok: f64,
    /// Extra decode ms per token per token of context (attention grows with context).
    pub ctx_penalty: f64,
    /// Relative jitter, e.g. 0.05 = ±5%.
    pub jitter: f64,
    /// Actually sleep so wall clock is real; tests turn this off.
    pub sleep: bool,
    calls: u64,
}

impl Default for Mock {
    fn default() -> Self {
        Self {
            prefill_ms_per_tok: 0.25,
            decode_ms_per_tok: 4.0,
            ctx_penalty: 0.002,
            jitter: 0.05,
            sleep: true,
            calls: 0,
        }
    }
}

impl Mock {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn instant() -> Self {
        Self {
            sleep: false,
            ..Self::default()
        }
    }
}

/// Small deterministic PRNG (xorshift64*), so the mock needs no extra crates.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Rough token estimate (~4 chars/token). Real tokenizers differ; the mock
/// only needs something monotonic in prompt length.
pub fn approx_tokens(s: &str) -> u64 {
    (s.chars().count() as u64).div_ceil(4).max(1)
}

const VOCAB: &[&str] = &[
    "the",
    "model",
    "loads",
    "weights",
    "and",
    "decodes",
    "one",
    "token",
    "at",
    "a",
    "time",
    "memory",
    "bandwidth",
    "limits",
    "throughput",
    "while",
    "prefill",
    "is",
    "compute",
    "bound",
    "cache",
    "context",
    "grows",
    "so",
    "attention",
    "costs",
    "more",
    "per",
    "step",
];

fn hash(s: &str) -> u64 {
    // FNV-1a
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl Mock {
    fn reply(&self, prompt: &str, p: &GenParams, rng: &mut Rng) -> String {
        let sampling = p.temperature > 0.0;
        let t = p.temperature.clamp(0.0, 2.0) as f64;

        // Critic protocol.
        if let Some(rest) = prompt
            .rsplit("PROPOSED ANSWER:")
            .next()
            .filter(|_| prompt.contains("PROPOSED ANSWER:"))
        {
            let proposed: Option<i64> = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            let truth = agent::lookup_answer(prompt);
            let correct = matches!((proposed, truth), (Some(a), Some(b)) if a == b);
            // Imperfect critic: catches 70% of wrong answers, wrongly rejects 10% of right ones.
            let says_yes = if correct {
                rng.f64() > 0.10
            } else {
                rng.f64() > 0.70
            };
            return if says_yes { "YES".into() } else { "NO".into() };
        }

        // Agent protocol.
        if prompt.contains("TASK:") && prompt.contains("CALC:") {
            if sampling && rng.f64() < 0.05 * t {
                return "I think the answer is probably around there.".into();
            }
            let Some(task) = agent::lookup_task(prompt) else {
                return "ANSWER: 0".into();
            };
            let history = prompt.rsplit("HISTORY:").next().unwrap_or("");
            let last_calc = history
                .lines()
                .filter(|l| l.trim_start().starts_with("CALC:") && !l.contains("ERROR"))
                .filter_map(|l| l.split_once("->").map(|(_, r)| r.trim().to_string()))
                .next_back();
            let rejected = history
                .lines()
                .next_back()
                .is_some_and(|l| l.contains("CRITIC: rejected"));
            return match last_calc {
                None => format!("CALC: {}", task.expr),
                Some(v) => {
                    let mut ans: i64 = v.parse().unwrap_or(0);
                    // Sampling sometimes slips; greedy is stable. A rejection makes it recheck.
                    if sampling && !rejected && rng.f64() < 0.35 * t {
                        ans += 1 + (rng.next_u64() % 9) as i64;
                    }
                    format!("ANSWER: {ans}")
                }
            };
        }

        // Plain completion: pseudo-text. Greedy is a pure function of the prompt.
        let mut words = Vec::with_capacity(p.max_tokens as usize);
        let mut h = hash(prompt);
        for _ in 0..p.max_tokens {
            let idx = if sampling {
                (rng.next_u64() % VOCAB.len() as u64) as usize
            } else {
                h = h
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((h >> 33) % VOCAB.len() as u64) as usize
            };
            words.push(VOCAB[idx]);
        }
        words.join(" ")
    }
}

impl Backend for Mock {
    fn name(&self) -> String {
        "mock (SIMULATED)".into()
    }

    fn generate(&mut self, prompt: &str, p: &GenParams) -> Result<Generation> {
        self.calls += 1;
        let t0 = Instant::now();
        let mut rng =
            Rng::new(p.seed.unwrap_or(0) ^ self.calls.wrapping_mul(0xA24B_AED4_963E_E407));
        let text = self.reply(prompt, p, &mut rng);

        let prompt_tokens = approx_tokens(prompt);
        let completion_tokens =
            if text.contains(' ') && !text.starts_with("CALC") && !text.starts_with("ANSWER") {
                text.split_whitespace().count() as u64
            } else {
                approx_tokens(&text)
            }
            .min(p.max_tokens as u64);

        let mut j = || 1.0 + self.jitter * (2.0 * rng.f64() - 1.0);
        let sampling_overhead = if p.temperature > 0.0 { 1.03 } else { 1.0 };
        let prefill_ms = prompt_tokens as f64 * self.prefill_ms_per_tok * j();
        let ctx = prompt_tokens as f64 + completion_tokens as f64 / 2.0;
        let decode_ms = completion_tokens as f64
            * (self.decode_ms_per_tok + self.ctx_penalty * ctx)
            * sampling_overhead
            * j();

        if self.sleep {
            std::thread::sleep(Duration::from_secs_f64((prefill_ms + decode_ms) / 1000.0));
        }
        let wall_ms = if self.sleep {
            t0.elapsed().as_secs_f64() * 1000.0
        } else {
            prefill_ms + decode_ms + 0.5
        };

        Ok(Generation {
            text,
            prompt_tokens,
            completion_tokens,
            prefill_ms: Some(prefill_ms),
            decode_ms: Some(decode_ms),
            load_ms: None,
            wall_ms,
            simulated: true,
            ..Default::default()
        })
    }

    fn model_info(&mut self) -> Option<String> {
        Some(format!(
            "simulator: prefill {:.2} ms/tok, decode {:.2} ms/tok (+{} ms/tok per ctx token), jitter ±{:.0}%",
            self.prefill_ms_per_tok,
            self.decode_ms_per_tok,
            self.ctx_penalty,
            self.jitter * 100.0
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic() {
        let mut m = Mock::instant();
        let p = GenParams {
            max_tokens: 16,
            ..Default::default()
        };
        let a = m.generate("hello", &p).unwrap();
        let b = m.generate("hello", &p).unwrap();
        assert_eq!(a.text, b.text);
        assert_eq!(a.completion_tokens, 16);
        assert!(a.simulated);
    }

    #[test]
    fn longer_prompt_costs_more_prefill() {
        let mut m = Mock::instant();
        m.jitter = 0.0;
        let p = GenParams::default();
        let short = m.generate("hi", &p).unwrap();
        let long = m.generate(&"word ".repeat(500), &p).unwrap();
        assert!(long.prefill_ms.unwrap() > short.prefill_ms.unwrap() * 10.0);
        assert!(long.decode_ms.unwrap() > short.decode_ms.unwrap());
    }
}
