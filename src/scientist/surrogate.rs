//! The surrogate: a transparent model that answers "what do we currently
//! believe configuration X costs?" from the fitted lines and the measured
//! agent configurations. Nothing is learned that cannot be printed.
//!
//!   call wall ≈ prefill(evaluated prompt tokens) + decode(output tokens) + overhead
//!   prefill(p) = prefill.fixed + prefill.per_token · p        (robust line)
//!   decode(n)  = decode.fixed  + decode.per_token  · n        (robust line)
//!
//! A prefix-cache hit is modelled as evaluating one prompt token, i.e. it
//! pays the fixed prefill cost only — the assumption the first real
//! cache-hit measurement forced.

use super::beliefs::AgentConfig;
use super::fit::{Est, Line};

#[derive(Debug, Clone)]
pub struct Surrogate {
    pub prefill: Option<Line>,
    pub decode: Option<Line>,
    pub overhead: Option<Est>,
    /// Median evaluated prompt tokens of the default prompt (no padding).
    pub base_prompt_tokens: f64,
    pub agent: Vec<AgentConfig>,
}

/// A single completion to price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallQuery {
    pub prompt_tokens: f64,
    pub max_tokens: f64,
    /// Prompt already in the KV cache (only ~1 token is evaluated).
    pub cache_hit: bool,
}

impl CallQuery {
    /// Parse "prompt_tokens=512 max_tokens=64 cache=true" (any order, any subset).
    pub fn parse(s: &str, base_prompt: f64) -> anyhow::Result<Self> {
        let mut q = CallQuery {
            prompt_tokens: base_prompt,
            max_tokens: 64.0,
            cache_hit: false,
        };
        for kv in s.split([' ', ',']).filter(|t| !t.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("expected key=value, got {kv:?}"))?;
            match k {
                "prompt_tokens" | "prompt" => q.prompt_tokens = v.parse()?,
                "max_tokens" | "max" | "out" => q.max_tokens = v.parse()?,
                "cache" | "cache_hit" | "cache_prompt" => q.cache_hit = v.parse()?,
                _ => anyhow::bail!("unknown key {k:?}; use prompt_tokens, max_tokens, cache"),
            }
        }
        Ok(q)
    }
}

/// Predicted cost of one call, by phase.
#[derive(Debug, Clone, Copy)]
pub struct CallPrediction {
    pub prefill_ms: Est,
    pub decode_ms: Est,
    pub overhead_ms: f64,
    pub wall_ms: Est,
}

impl Surrogate {
    pub fn evaluated_tokens(q: &CallQuery) -> f64 {
        if q.cache_hit { 1.0 } else { q.prompt_tokens }
    }

    /// Prefill and decode share bootstrap draws index-wise, so the wall
    /// interval reflects both fits' uncertainty at once.
    pub fn call(&self, q: &CallQuery) -> Option<CallPrediction> {
        let (pf, dc) = (self.prefill.as_ref()?, self.decode.as_ref()?);
        let p = Self::evaluated_tokens(q);
        let ovh = self.overhead.map(|o| o.value.max(0.0)).unwrap_or(0.0);
        let n = pf.draws.len().min(dc.draws.len());
        let mut walls: Vec<f64> = (0..n)
            .map(|i| {
                let (a, b) = pf.draws[i];
                let (c, d) = dc.draws[i];
                (a + b * p).max(0.0) + (c + d * q.max_tokens).max(0.0) + ovh
            })
            .collect();
        walls.sort_by(|a, b| a.total_cmp(b));
        let prefill = pf.at_ci(p);
        let decode = dc.at_ci(q.max_tokens);
        let point = prefill.value.max(0.0) + decode.value.max(0.0) + ovh;
        let (lo, hi) = if walls.is_empty() {
            (point, point)
        } else {
            (
                crate::stats::percentile(&walls, 0.025),
                crate::stats::percentile(&walls, 0.975),
            )
        };
        Some(CallPrediction {
            prefill_ms: prefill,
            decode_ms: decode,
            overhead_ms: ovh,
            wall_ms: Est {
                value: point,
                lo,
                hi,
            },
        })
    }

    pub fn prefill_at(&self, tokens: f64) -> Option<Est> {
        Some(self.prefill.as_ref()?.at_ci(tokens))
    }

    pub fn decode_tok_s(&self) -> Option<f64> {
        let d = self.decode.as_ref()?;
        (d.slope.value > 0.0).then(|| 1000.0 / d.slope.value)
    }

    pub fn agent_config(&self, key: &str) -> Option<&AgentConfig> {
        self.agent.iter().find(|c| c.key == key)
    }

    /// Share of an agent call's time spent in prefill (what prompt-side
    /// levers can touch), at the baseline agent's measured prompt size.
    pub fn agent_prefill_share(&self) -> Option<f64> {
        let base = self.agent.iter().find(|c| c.key == super::BASELINE_KEY)?;
        let c = self.call(&CallQuery {
            prompt_tokens: base.prompt_per_call,
            max_tokens: 10.0,
            cache_hit: false,
        })?;
        (c.wall_ms.value > 0.0).then(|| c.prefill_ms.value.max(0.0) / c.wall_ms.value)
    }

    pub fn print_query(&self, q: &CallQuery) {
        match self.call(q) {
            None => println!(
                "Can't price that yet: need fitted prefill and decode lines (run a prompt-tokens and a max-tokens sweep)."
            ),
            Some(c) => {
                println!(
                    "what-if: prompt {} tokens{}, {} output tokens",
                    q.prompt_tokens,
                    if q.cache_hit {
                        " (cache hit → 1 token evaluated)"
                    } else {
                        ""
                    },
                    q.max_tokens
                );
                let f = |e: Est| format!("{:.0} ms [{:.0}, {:.0}]", e.value, e.lo, e.hi);
                println!("  prefill  {}", f(c.prefill_ms));
                println!("  decode   {}", f(c.decode_ms));
                println!("  overhead {:.1} ms", c.overhead_ms);
                println!(
                    "  wall     {}   (95% interval from bootstrap draws of both lines)",
                    f(c.wall_ms)
                );
                let p = Self::evaluated_tokens(q);
                for (name, line, x) in [
                    ("prompt", self.prefill.as_ref(), p),
                    ("output", self.decode.as_ref(), q.max_tokens),
                ] {
                    if let Some(l) = line
                        && (x < l.x_min * 0.9 || x > l.x_max * 1.1)
                    {
                        println!(
                            "  ⚠ {name} length {x} is outside the measured range [{}, {}]: this is an extrapolation.",
                            l.x_min, l.x_max
                        );
                    }
                }
            }
        }
    }
}
