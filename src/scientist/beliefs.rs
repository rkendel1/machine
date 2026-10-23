//! The explicit belief state: every quantity the scientist thinks it knows,
//! with an interval, a sample size, the method that produced it, and a
//! status. Serialized to `beliefs.json` next to the notebook so it can be
//! diffed, argued with, and checked into a repo.

use super::fit::Est;
use crate::metrics;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Interval is tight (parameters) or excludes zero (effects).
    Known,
    /// Measured, but the interval is wide or includes zero.
    Uncertain,
    /// No data that could estimate it.
    Untested,
}

impl Status {
    pub fn tag(self) -> &'static str {
        match self {
            Status::Known => "KNOWN    ",
            Status::Uncertain => "UNCERTAIN",
            Status::Untested => "UNTESTED ",
        }
    }
}

/// A parameter (e.g. ms per prompt token) or an effect (a difference
/// between two settings of one knob).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub key: String,
    /// One-line plain-English meaning.
    pub what: String,
    pub est: Option<Est>,
    pub unit: String,
    /// Observations behind the estimate.
    pub n: usize,
    pub status: Status,
    /// How it was estimated (so a human can disagree with the method).
    pub method: String,
    /// True for effects (status from "excludes zero"), false for parameters.
    pub effect: bool,
}

/// Parameters count as known once the 95% half-width is within this
/// fraction of the estimate.
pub const KNOWN_REL_WIDTH: f64 = 0.15;

impl Belief {
    pub fn new(key: &str, what: &str, unit: &str, effect: bool, method: &str) -> Self {
        Self {
            key: key.into(),
            what: what.into(),
            est: None,
            unit: unit.into(),
            n: 0,
            status: Status::Untested,
            method: method.into(),
            effect,
        }
    }

    pub fn with(mut self, est: Option<Est>, n: usize) -> Self {
        self.n = n;
        self.status = match est {
            None => Status::Untested,
            Some(e) if self.effect && e.excludes_zero() => Status::Known,
            Some(e) if !self.effect && e.rel_width() <= KNOWN_REL_WIDTH => Status::Known,
            Some(_) => Status::Uncertain,
        };
        self.est = est;
        self
    }

    /// Relative uncertainty in [0, 1]: 1 = know nothing, 0 = pinned down.
    /// Effects are scaled by |estimate| + half-width so a tight null result
    /// ("no effect") counts as known, not as infinitely uncertain.
    pub fn uncertainty(&self) -> f64 {
        match self.est {
            None => 1.0,
            Some(e) if !e.lo.is_finite() || !e.hi.is_finite() => 1.0,
            Some(e) if self.effect => {
                let scale = e.value.abs() + e.half_width();
                if scale <= 0.0 {
                    0.0
                } else {
                    (e.half_width() / scale).min(1.0)
                }
            }
            Some(e) => e.rel_width().min(1.0),
        }
    }

    pub fn fmt_est(&self) -> String {
        match self.est {
            None => "—".into(),
            Some(e) => {
                let f = |v: f64| fmt_unit(&self.unit, v);
                let ci = if e.lo.is_finite() && e.hi.is_finite() {
                    format!("[{}, {}]", f(e.lo), f(e.hi))
                } else {
                    "[n=1, no interval]".into()
                };
                format!("{} {ci}", f(e.value))
            }
        }
    }
}

fn fmt_unit(unit: &str, v: f64) -> String {
    match unit {
        "pts" => format!("{:+.0} pts", v * 100.0),
        "%" => format!("{:.1}%", v * 100.0),
        "rate" => format!("{:.0}%", v * 100.0),
        u if u.starts_with('Δ') => {
            format!("{}{}", if v >= 0.0 { "+" } else { "" }, short(v, &u[2..]))
        }
        u => short(v, u),
    }
}

fn short(v: f64, unit: &str) -> String {
    let n = metrics::short(v);
    if unit.is_empty() {
        n
    } else {
        format!("{n} {unit}")
    }
}

/// Measured behaviour of one agent configuration (full task set only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// e.g. "critic=none window=all cache=false temp=0"
    pub key: String,
    pub critic: String,
    pub memory_window: String,
    pub cache_prompt: bool,
    pub temperature: f32,
    pub episodes: usize,
    /// Separate runs (groups) pooled into this config.
    pub runs: usize,
    pub success: Est,
    pub calls: Est,
    pub wall_ms: Est,
    pub prompt_per_call: f64,
    pub format_errors_per_task: f64,
    pub critic_false_accepts: f64,
}

/// An assumption the data contradicts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub key: String,
    pub title: String,
    /// The numbers that show it.
    pub evidence: String,
    /// What to do differently because of it.
    pub consequence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    pub mode: String,
    pub rule: String,
    pub episodes: usize,
    /// Share of all failed episodes.
    pub share: f64,
    pub configs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    pub version: u32,
    pub backend: String,
    pub simulated: bool,
    pub fitted_at: String,
    pub run_rows: usize,
    pub agent_episodes: usize,
    pub beliefs: Vec<Belief>,
    pub agent_configs: Vec<AgentConfig>,
    pub anomalies: Vec<Anomaly>,
    pub failure_modes: Vec<FailureMode>,
    /// Tasks that fail in (nearly) every configuration: not a knob problem.
    pub systematic_failures: Vec<String>,
    /// Modeling choices, stated so they can be challenged.
    pub methods: Vec<String>,
}

impl BeliefState {
    pub fn get(&self, key: &str) -> Option<&Belief> {
        self.beliefs.iter().find(|b| b.key == key)
    }

    pub fn path(dir: &std::path::Path) -> PathBuf {
        dir.join("beliefs.json")
    }

    pub fn save(&self, dir: &std::path::Path) -> Result<PathBuf> {
        let p = Self::path(dir);
        std::fs::write(&p, serde_json::to_string_pretty(self)?)?;
        Ok(p)
    }

    pub fn print(&self) {
        println!("═══ BELIEFS ═══  {}", self.backend);
        println!(
            "fitted {} from {} run rows and {} agent episodes{}",
            self.fitted_at,
            self.run_rows,
            self.agent_episodes,
            if self.simulated { "  [SIMULATED]" } else { "" }
        );
        println!(
            "status: KNOWN = interval within ±{:.0}% (parameters) or excluding zero (effects)\n",
            KNOWN_REL_WIDTH * 100.0
        );
        for b in &self.beliefs {
            println!(
                "  {}  {:<34} {:<40} n={:<4}",
                b.status.tag(),
                b.key,
                b.fmt_est(),
                b.n
            );
            println!("             {}  ({})", b.what, b.method);
        }
        if !self.agent_configs.is_empty() {
            println!("\n  agent configurations (10-task runs only):");
            println!(
                "    {:<44} {:>4} {:>18} {:>16} {:>22}",
                "config", "eps", "success [95%]", "calls [95%]", "wall/task [95%]"
            );
            for c in &self.agent_configs {
                println!(
                    "    {:<44} {:>4} {:>18} {:>16} {:>22}",
                    c.key,
                    c.episodes,
                    format!(
                        "{:.0}% [{:.0}, {:.0}]",
                        c.success.value * 100.0,
                        c.success.lo * 100.0,
                        c.success.hi * 100.0
                    ),
                    format!(
                        "{:.1} [{:.1}, {:.1}]",
                        c.calls.value, c.calls.lo, c.calls.hi
                    ),
                    format!(
                        "{:.0} [{:.0}, {:.0}] ms",
                        c.wall_ms.value, c.wall_ms.lo, c.wall_ms.hi
                    ),
                );
            }
        }
    }
}
