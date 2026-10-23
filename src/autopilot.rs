//! Autopilot: the predict → run → score → learn cycle without a human typing
//! each command.
//!
//! It is a careful lab assistant, not an open-ended agent:
//! - the proposer is deterministic (no LLM) and reads only the notebook;
//! - every proposal is exactly one single-variable change, expressible as a
//!   `machine run|sweep|agent` command a human could have typed (`cli()`);
//! - every prediction is a number derived from earlier measurements and is
//!   scored by the same `predict::score` the human path uses;
//! - load-time knobs (quant, threads, GPU layers, context) are never changed.

use crate::advisor::{self, Lever};
use crate::agent::{self, CriticMode};
use crate::backend::GenParams;
use crate::backend::mock::approx_tokens;
use crate::experiment::{self, SweepVar};
use crate::lab::{Lab, Outcome};
use crate::learnings::{self, Learning};
use crate::metrics;
use crate::notebook::now_secs;
use crate::notebook::{Kind, Record};
use crate::predict::Prediction;
use crate::stats::{self, Summary};
use anyhow::{Result, bail};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Focus {
    Prefill,
    Decode,
    Agent,
    Quant,
    Critic,
    Cache,
}

/// The agent knobs the autopilot may change — all per-request, none load-time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentSettings {
    pub critic: CriticMode,
    pub memory_window: Option<usize>,
    pub cache_prompt: bool,
    pub temperature: f32,
}

impl AgentSettings {
    pub const BASELINE: Self = Self {
        critic: CriticMode::None,
        memory_window: None,
        cache_prompt: false,
        temperature: 0.0,
    };

    /// Names of the settings that differ from `other`.
    pub fn diff(&self, other: &Self) -> Vec<&'static str> {
        let mut d = Vec::new();
        if self.critic != other.critic {
            d.push("critic");
        }
        if self.memory_window != other.memory_window {
            d.push("memory_window");
        }
        if self.cache_prompt != other.cache_prompt {
            d.push("cache_prompt");
        }
        if self.temperature != other.temperature {
            d.push("temperature");
        }
        d
    }

    pub fn critic_tag(&self) -> String {
        format!("{:?}", self.critic).to_lowercase()
    }

    pub fn window_tag(&self) -> String {
        self.memory_window
            .map(|w| w.to_string())
            .unwrap_or("all".into())
    }

    /// Does an agent notebook row use exactly these settings?
    pub fn matches(&self, r: &Record) -> bool {
        r.kind == Kind::Agent
            && r.tags.get("critic").map(String::as_str) == Some(self.critic_tag().as_str())
            && r.tags.get("memory_window").map(String::as_str) == Some(self.window_tag().as_str())
            && r.params.as_ref().is_some_and(|p| {
                p.cache_prompt == self.cache_prompt && p.temperature == self.temperature
            })
    }
}

/// One experiment, in terms of the existing commands.
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentSpec {
    /// `machine run --repeat N --warmup W --max-tokens M --ignore-eos`
    Run {
        repeat: usize,
        warmup: usize,
        max_tokens: u32,
    },
    /// `machine sweep --var V --values a,b,.. --repeat N --warmup W --max-tokens M --ignore-eos --metric X`
    Sweep {
        var: SweepVar,
        values: Vec<String>,
        repeat: usize,
        warmup: usize,
        max_tokens: u32,
        metric: &'static str,
    },
    /// `machine agent [--critic ..] [--memory-window ..] [--cache-prompt] [--temperature ..]`
    Agent {
        settings: AgentSettings,
        /// The settings this run is compared against (equal for a baseline).
        baseline: AgentSettings,
        tasks: Vec<usize>,
    },
}

impl ExperimentSpec {
    /// Reject anything that is not a single-variable experiment.
    pub fn validate(&self) -> Result<()> {
        match self {
            ExperimentSpec::Run { repeat, .. } => {
                if *repeat < 3 {
                    bail!("a replication needs ≥3 runs to estimate spread");
                }
            }
            ExperimentSpec::Sweep {
                values,
                metric,
                repeat,
                ..
            } => {
                let distinct: BTreeSet<&String> = values.iter().collect();
                if distinct.len() < 2 || distinct.len() != values.len() {
                    bail!("a sweep needs ≥2 distinct values of one variable");
                }
                if !metrics::is_known(metric) {
                    bail!("unknown metric {metric}");
                }
                if *repeat < 2 {
                    bail!("a sweep needs ≥2 runs per point to tell signal from noise");
                }
            }
            ExperimentSpec::Agent {
                settings,
                baseline,
                tasks,
            } => {
                if tasks.is_empty() || tasks.iter().any(|t| *t >= agent::TASKS.len()) {
                    bail!("agent tasks out of range");
                }
                if settings.diff(baseline).len() > 1 {
                    bail!(
                        "agent spec changes {:?} at once — only one variable allowed",
                        settings.diff(baseline)
                    );
                }
            }
        }
        Ok(())
    }

    /// The equivalent command line, minus backend and predictions.
    pub fn cli(&self) -> String {
        match self {
            ExperimentSpec::Run {
                repeat,
                warmup,
                max_tokens,
            } => format!(
                "machine run --repeat {repeat} --warmup {warmup} --max-tokens {max_tokens} --ignore-eos"
            ),
            ExperimentSpec::Sweep {
                var,
                values,
                repeat,
                warmup,
                max_tokens,
                metric,
            } => format!(
                "machine sweep --var {} --values {} --repeat {repeat} --warmup {warmup} --max-tokens {max_tokens} --ignore-eos --metric {metric}",
                var.cli_name(),
                values.join(",")
            ),
            ExperimentSpec::Agent {
                settings, tasks, ..
            } => {
                let mut s = String::from("machine agent");
                if *tasks != all_tasks() {
                    let t: Vec<String> = tasks.iter().map(|t| t.to_string()).collect();
                    s += &format!(" --tasks {}", t.join(","));
                }
                if settings.critic != CriticMode::None {
                    s += &format!(" --critic {}", settings.critic_tag());
                }
                if let Some(w) = settings.memory_window {
                    s += &format!(" --memory-window {w}");
                }
                if settings.cache_prompt {
                    s += " --cache-prompt";
                }
                if settings.temperature != 0.0 {
                    s += &format!(" --temperature {}", settings.temperature);
                }
                s
            }
        }
    }
}

pub fn all_tasks() -> Vec<usize> {
    (0..agent::TASKS.len()).collect()
}

/// A proposed trial: one spec, its numeric predictions, and why.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Dedupe key: a lever is tried at most once per session (replications are numbered).
    pub key: String,
    pub spec: ExperimentSpec,
    pub predictions: Vec<Prediction>,
    /// The notebook numbers behind the proposal and predictions.
    pub rationale: String,
    pub lesson: u8,
    pub label: String,
    /// Expected wall time, from the cost model.
    pub est_ms: f64,
}

impl Proposal {
    /// Full command a human could type, including the predictions.
    pub fn command(&self, backend_flags: &str) -> String {
        // Flags go after the subcommand: `machine sweep --backend mock ...`.
        let cli = self.spec.cli();
        let rest = cli.strip_prefix("machine ").unwrap_or(&cli);
        let (cmd, args) = rest.split_once(' ').unwrap_or((rest, ""));
        let mut s = format!("machine {cmd} {backend_flags}{args}")
            .trim_end()
            .to_string();
        s += &format!(" --lesson {} --label {}", self.lesson, self.label);
        for p in &self.predictions {
            let v: Vec<String> = p.values.iter().map(|v| fmt_pred(*v)).collect();
            s += &format!(" --predict {}={}", p.metric, v.join(","));
        }
        s
    }
}

fn fmt_pred(v: f64) -> String {
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Per-token costs measured so far on this backend. Falls back to stated,
/// pessimistic priors when nothing is measured yet (`measured = false`).
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    pub decode_ms_per_tok: f64,
    pub prefill_ms_per_tok: f64,
    pub base_prompt_tokens: f64,
    pub overhead_ms: f64,
    pub measured: bool,
}

impl CostModel {
    pub const PRIOR: Self = Self {
        decode_ms_per_tok: 40.0,
        prefill_ms_per_tok: 7.0,
        base_prompt_tokens: 30.0,
        overhead_ms: 5.0,
        measured: false,
    };

    pub fn from_records(records: &[Record]) -> Self {
        let rows: Vec<&Record> = records
            .iter()
            .filter(|r| r.kind == Kind::Run && !r.warmup)
            .filter(|r| r.params.as_ref().is_none_or(|p| !p.cache_prompt))
            .collect();
        let ratio = |num: &str, den: &str, rows: &[&Record]| -> Option<f64> {
            let xs: Vec<f64> = rows
                .iter()
                .filter_map(|r| {
                    let d = *r.metrics.get(den)?;
                    (d >= 1.0).then(|| r.metrics.get(num).map(|n| n / d))?
                })
                .collect();
            Summary::of(&xs).map(|s| s.p50)
        };
        let decode = ratio("decode_ms", "completion_tokens", &rows);
        // Prefill per token from the longest prompts we have: short prompts
        // overstate it (fixed per-call cost spread over few tokens).
        let mut by_len = rows.clone();
        by_len.sort_by(|a, b| {
            let pa = a.metrics.get("prompt_tokens").copied().unwrap_or(0.0);
            let pb = b.metrics.get("prompt_tokens").copied().unwrap_or(0.0);
            pb.total_cmp(&pa)
        });
        let top = &by_len[..by_len.len().min(10)];
        let prefill = ratio("prefill_ms", "prompt_tokens", top);
        let base_rows: Vec<&Record> = rows.iter().copied().filter(|r| r.var.is_none()).collect();
        let median = |k: &str, rows: &[&Record]| -> Option<f64> {
            let xs: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.metrics.get(k).copied())
                .collect();
            Summary::of(&xs).map(|s| s.p50)
        };
        Self {
            decode_ms_per_tok: decode.unwrap_or(Self::PRIOR.decode_ms_per_tok),
            prefill_ms_per_tok: prefill.unwrap_or(Self::PRIOR.prefill_ms_per_tok),
            base_prompt_tokens: median("prompt_tokens", &base_rows)
                .unwrap_or(Self::PRIOR.base_prompt_tokens),
            overhead_ms: median("overhead_ms", &rows)
                .unwrap_or(Self::PRIOR.overhead_ms)
                .max(0.0),
            measured: decode.is_some(),
        }
    }

    pub fn call_ms(&self, prompt_tokens: f64, out_tokens: f64) -> f64 {
        self.overhead_ms
            + self.prefill_ms_per_tok * prompt_tokens
            + self.decode_ms_per_tok * out_tokens
    }

    pub fn decode_tok_s(&self) -> f64 {
        1000.0 / self.decode_ms_per_tok
    }

    /// Which phase dominates a default 64-token call.
    pub fn bottleneck(&self) -> String {
        let pre = self.prefill_ms_per_tok * self.base_prompt_tokens;
        let dec = self.decode_ms_per_tok * 64.0;
        let total = pre + dec + self.overhead_ms;
        if dec >= pre {
            format!(
                "decode-bound: {:.0}% of a 64-token completion is decode ({:.1} ms/token)",
                100.0 * dec / total,
                self.decode_ms_per_tok
            )
        } else {
            format!(
                "prefill-bound: {:.0}% of a 64-token completion is prefill ({:.1} ms/prompt token)",
                100.0 * pre / total,
                self.prefill_ms_per_tok
            )
        }
    }
}

/// Measured behaviour of one agent configuration.
#[derive(Debug, Clone, Copy)]
pub struct AgentStats {
    pub success: f64,
    pub calls: f64,
    pub wall_ms: f64,
    pub prompt_per_call: f64,
    pub episodes: usize,
}

impl AgentStats {
    /// Latest group of episodes run with `settings` over at least `n_tasks` tasks.
    pub fn from_records(
        records: &[Record],
        settings: &AgentSettings,
        n_tasks: usize,
    ) -> Option<Self> {
        let latest_group = records
            .iter()
            .rev()
            .filter(|r| settings.matches(r))
            .map(|r| r.group.clone())
            .find(|g| {
                records
                    .iter()
                    .filter(|r| r.kind == Kind::Agent && &r.group == g)
                    .count()
                    >= n_tasks
            })?;
        let rows: Vec<&Record> = records
            .iter()
            .filter(|r| r.kind == Kind::Agent && r.group == latest_group)
            .collect();
        let mean = |k: &str| {
            Summary::of(
                &rows
                    .iter()
                    .filter_map(|r| r.metrics.get(k).copied())
                    .collect::<Vec<_>>(),
            )
            .map(|s| s.mean)
        };
        let calls = mean("calls")?;
        Some(Self {
            success: mean("success")?,
            calls,
            wall_ms: mean("wall_ms")?,
            prompt_per_call: mean("prompt_tokens")? / calls.max(1.0),
            episodes: rows.len(),
        })
    }

    /// Prior for an agent that was never run: two calls (CALC, ANSWER) plus
    /// one format error on average, success at maximum uncertainty (50%).
    pub fn prior(cost: &CostModel) -> Self {
        let prompt = approx_tokens(&agent::actor_prompt(&agent::TASKS[0], &[], None)) as f64;
        let calls = 3.0;
        Self {
            success: 0.5,
            calls,
            wall_ms: calls * cost.call_ms(prompt, 10.0),
            prompt_per_call: prompt,
            episodes: 0,
        }
    }
}

/// What the autopilot carries between trials.
#[derive(Debug, Clone)]
pub struct AutopilotState {
    pub tried: BTreeSet<String>,
    pub budget_ms: f64,
    pub spent_ms: f64,
    pub steps: usize,
    pub max_steps: usize,
    pub focus: Option<Focus>,
    pub bottleneck: String,
    /// Most recent learnings, oldest first (keeps the last 10).
    pub recent: Vec<String>,
    /// Trials this session whose predictions missed, by key → (spec, measured means per point).
    pub missed: Vec<(String, ExperimentSpec, Vec<Prediction>)>,
}

impl AutopilotState {
    pub fn new(budget_min: f64, max_steps: usize, focus: Option<Focus>) -> Self {
        Self {
            tried: BTreeSet::new(),
            budget_ms: budget_min * 60_000.0,
            spent_ms: 0.0,
            steps: 0,
            max_steps,
            focus,
            bottleneck: String::new(),
            recent: Vec::new(),
            missed: Vec::new(),
        }
    }

    pub fn remaining_ms(&self) -> f64 {
        (self.budget_ms - self.spent_ms).max(0.0)
    }

    pub fn remember(&mut self, learning: String) {
        self.recent.push(learning);
        if self.recent.len() > 10 {
            self.recent.remove(0);
        }
    }
}

const MAX_REPLICATIONS: usize = 2;

fn lever_key(l: &Lever, nth_replication: usize) -> String {
    match l {
        Lever::Baseline => "baseline".into(),
        Lever::NoiseFloor { .. } => format!("noise#{nth_replication}"),
        Lever::Sweep(v) => format!("sweep:{v}"),
        Lever::QuantNeedsRestart => "quant".into(),
        Lever::AgentBaseline => "agent:baseline".into(),
        Lever::Critic(m) => format!("agent:critic={m}"),
        Lever::MemoryWindow => "agent:memory_window=1".into(),
        Lever::AgentCache => "agent:cache_prompt".into(),
    }
}

fn in_focus(l: &Lever, focus: Option<Focus>) -> bool {
    let Some(f) = focus else { return true };
    match l {
        // Prerequisites for any prediction: always allowed.
        Lever::Baseline | Lever::NoiseFloor { .. } => true,
        Lever::Sweep("prompt_tokens") => f == Focus::Prefill,
        Lever::Sweep("cache_prompt") => matches!(f, Focus::Cache | Focus::Prefill),
        Lever::Sweep(_) => f == Focus::Decode,
        Lever::QuantNeedsRestart => f == Focus::Quant,
        Lever::AgentBaseline => matches!(f, Focus::Agent | Focus::Critic | Focus::Cache),
        Lever::Critic(_) => f == Focus::Critic,
        Lever::MemoryWindow => f == Focus::Agent,
        Lever::AgentCache => matches!(f, Focus::Cache | Focus::Agent),
    }
}

/// Levers from the notebook, plus those that would become available once
/// already-tried prerequisites land (so a dry run can show the full sequence).
fn candidate_levers(records: &[Record], tried: &BTreeSet<String>) -> Vec<Lever> {
    let mut ls = advisor::levers(records);
    if ls == [Lever::Baseline] && tried.contains("baseline") {
        // The baseline is itself a repeat group, so the noise floor will be
        // known; only a noisy result (seen in a real run) re-triggers it.
        ls.extend(advisor::SWEEP_ORDER.iter().map(|v| Lever::Sweep(v)));
        ls.push(Lever::AgentBaseline);
    }
    if tried.contains("agent:baseline") && !records.iter().any(|r| r.kind == Kind::Agent) {
        ls.extend([
            Lever::Critic("model"),
            Lever::Critic("oracle"),
            Lever::MemoryWindow,
            Lever::AgentCache,
        ]);
    }
    ls
}

/// Propose exactly one next trial, or None when nothing high-value remains.
/// Pure and deterministic: same notebook + same state → same proposal.
pub fn propose_next(state: &AutopilotState, records: &[Record]) -> Option<Proposal> {
    let cost = CostModel::from_records(records);
    let replications = state
        .tried
        .iter()
        .filter(|k| k.starts_with("noise#"))
        .count();
    for lever in candidate_levers(records, &state.tried) {
        if !in_focus(&lever, state.focus) || lever == Lever::QuantNeedsRestart {
            continue;
        }
        let key = lever_key(&lever, replications + 1);
        if let Lever::NoiseFloor { cv } = lever {
            // Unknown noise needs one repeat group; a noisy one gets up to two more.
            let cap = if cv.is_none() { 1 } else { MAX_REPLICATIONS };
            if replications >= cap {
                continue;
            }
        } else if state.tried.contains(&key) {
            continue;
        }
        if let Some(p) = build(lever, key, &cost, records) {
            return Some(p);
        }
    }
    // Fallback: replicate a trial whose prediction missed, predicting that
    // the measured result reproduces. This reduces uncertainty, not novelty.
    for (key, spec, measured) in &state.missed {
        let ckey = format!("confirm:{key}");
        if state.tried.contains(&ckey) || key.starts_with("confirm:") {
            continue;
        }
        let est = estimate_ms(spec, &cost, records);
        return Some(Proposal {
            key: ckey,
            spec: spec.clone(),
            predictions: measured.clone(),
            rationale: format!(
                "`{key}` missed its prediction. Re-running it unchanged, predicting the measured values reproduce, separates a wrong model from a noisy measurement."
            ),
            lesson: 2,
            label: format!("ap-confirm-{}", key.replace([':', '=', '#'], "-")),
            est_ms: est,
        });
    }
    None
}

fn build(lever: Lever, key: String, cost: &CostModel, records: &[Record]) -> Option<Proposal> {
    let prior_note = if cost.measured {
        ""
    } else {
        " (no measurements yet: predictions use stated priors of 40 ms/token decode, 7 ms/token prefill)"
    };
    let base_call = cost.call_ms(cost.base_prompt_tokens, 64.0);
    let label = format!("ap-{}", key.replace([':', '=', '#'], "-"));
    let (spec, predictions, rationale, lesson) = match lever {
        Lever::Baseline => (
            ExperimentSpec::Run {
                repeat: 5,
                warmup: 1,
                max_tokens: 64,
            },
            vec![
                pred("decode_tok_s", vec![cost.decode_tok_s()]),
                pred("wall_ms", vec![base_call]),
            ],
            format!(
                "No measured runs on this backend. Establish the baseline: 5 runs × 64 tokens after one warm-up{prior_note}."
            ),
            1,
        ),
        Lever::NoiseFloor { cv } => (
            ExperimentSpec::Run {
                repeat: 10,
                warmup: 2,
                max_tokens: 64,
            },
            vec![
                pred("decode_tok_s", vec![cost.decode_tok_s()]),
                pred("wall_ms", vec![base_call]),
            ],
            match cv {
                None => {
                    "No repeat group yet, so the noise floor is unknown. 10 runs after 2 warm-ups."
                        .into()
                }
                Some(cv) => format!(
                    "Latest repeat group has decode cv {:.1}% (>5%): effects under ~{:.0}% are invisible. Replicate with more warm-up before changing anything.",
                    cv * 100.0,
                    cv * 200.0
                ),
            },
            2,
        ),
        Lever::Sweep(var) => {
            let v = SweepVar::from_name(var)?;
            let (values, max_tokens, repeat, metric): (Vec<&str>, u32, usize, &'static str) =
                match v {
                    SweepVar::MaxTokens => (vec!["16", "32", "64"], 64, 3, "wall_ms"),
                    SweepVar::PromptTokens => (vec!["64", "256", "512"], 8, 3, "prefill_ms"),
                    SweepVar::Temperature => (vec!["0", "0.8"], 64, 5, "decode_tok_s"),
                    SweepVar::CachePrompt => (vec!["false", "true"], 8, 5, "prefill_ms"),
                    SweepVar::Seed => return None,
                };
            let predicted: Vec<f64> = values
                .iter()
                .map(|x| match v {
                    SweepVar::MaxTokens => {
                        cost.call_ms(cost.base_prompt_tokens, x.parse().unwrap_or(0.0))
                    }
                    SweepVar::PromptTokens => {
                        cost.prefill_ms_per_tok * x.parse::<f64>().unwrap_or(0.0)
                    }
                    SweepVar::Temperature => cost.decode_tok_s(),
                    // A cache hit re-evaluates at least one token.
                    SweepVar::CachePrompt if *x == "true" => cost.prefill_ms_per_tok,
                    _ => cost.prefill_ms_per_tok * cost.base_prompt_tokens,
                })
                .collect();
            let why = match v {
                SweepVar::MaxTokens => format!(
                    "Output length was never swept. Predicted from {:.1} ms/token decode + {:.0} ms fixed cost.",
                    cost.decode_ms_per_tok,
                    cost.call_ms(cost.base_prompt_tokens, 0.0)
                ),
                SweepVar::PromptTokens => format!(
                    "Prefill vs prompt length was never measured; agents grow their prompt every step. Predicted at {:.2} ms/prompt token.",
                    cost.prefill_ms_per_tok
                ),
                SweepVar::Temperature => "Greedy vs sampling speed was never compared. Null-hypothesis prediction: sampling costs nothing measurable in decode tok/s.".into(),
                _ => format!(
                    "Prefix caching is the top untried lever. With an identical prompt, a cache hit should re-evaluate ~1 token: prefill {:.0} ms → {:.1} ms.",
                    cost.prefill_ms_per_tok * cost.base_prompt_tokens,
                    cost.prefill_ms_per_tok
                ),
            };
            (
                ExperimentSpec::Sweep {
                    var: v,
                    values: values.iter().map(|s| s.to_string()).collect(),
                    repeat,
                    warmup: 1,
                    max_tokens,
                    metric,
                },
                vec![pred(metric, predicted)],
                format!("{why}{prior_note}"),
                3,
            )
        }
        Lever::QuantNeedsRestart => return None,
        Lever::AgentBaseline => {
            let s = AgentStats::prior(cost);
            (
                agent_spec(AgentSettings::BASELINE),
                agent_preds(&s),
                format!(
                    "No agent episodes yet. Prior: {:.0} calls/task × {:.0} ms/call (≈{:.0} prompt tokens each), success 50% (maximum uncertainty).",
                    s.calls,
                    s.wall_ms / s.calls,
                    s.prompt_per_call
                ),
                4,
            )
        }
        Lever::Critic(_) | Lever::MemoryWindow | Lever::AgentCache => {
            let base =
                AgentStats::from_records(records, &AgentSettings::BASELINE, agent::TASKS.len())
                    .unwrap_or_else(|| AgentStats::prior(cost));
            let mut s = base;
            let mut settings = AgentSettings::BASELINE;
            let (why, lesson) = match lever {
                Lever::Critic("model") => {
                    settings.critic = CriticMode::Model;
                    // One YES/NO call per answer: full prefill, ≤3 decode tokens.
                    let critic_call = cost.call_ms(base.prompt_per_call, 2.0);
                    s.calls += 1.0;
                    s.wall_ms += critic_call;
                    (
                        format!(
                            "No model critic yet. Prediction: +1 call/task (+{critic_call:.0} ms, mostly prefill), accuracy unchanged at {:.0}% (a model rarely catches its own errors).",
                            base.success * 100.0
                        ),
                        5,
                    )
                }
                Lever::Critic(_) => {
                    settings.critic = CriticMode::Oracle;
                    let retries = 1.0 - base.success;
                    s.wall_ms *= (base.calls + retries) / base.calls.max(1.0);
                    s.calls += retries;
                    (
                        format!(
                            "No oracle yet: it is the ceiling for any critic. Prediction: +{retries:.2} calls/task (one retry per wrong task), accuracy unchanged — retries under greedy decoding tend to repeat the same plan."
                        ),
                        5,
                    )
                }
                Lever::MemoryWindow => {
                    settings.memory_window = Some(1);
                    ("Every agent run used full history. Null-hypothesis prediction: a 1-line window changes nothing when episodes are ~2–3 steps.".to_string(), 6)
                }
                _ => {
                    settings.cache_prompt = true;
                    // Consecutive prompts share everything up to the newest
                    // history line, so ~25 tokens are re-evaluated per call.
                    let saved =
                        s.calls * (base.prompt_per_call - 25.0).max(0.0) * cost.prefill_ms_per_tok;
                    s.wall_ms = (s.wall_ms - saved).max(s.calls * cost.call_ms(25.0, 8.0));
                    (
                        format!(
                            "The agent re-sends ~{:.0} prompt tokens per call with an unchanging prefix. Prediction: prefix caching saves ~{saved:.0} ms/task (prefill of {:.0} → ~25 tokens per call); success and calls unchanged.",
                            base.prompt_per_call, base.prompt_per_call
                        ),
                        6,
                    )
                }
            };
            let mut spec = agent_spec(settings);
            if let ExperimentSpec::Agent { baseline, .. } = &mut spec {
                *baseline = AgentSettings::BASELINE;
            }
            let src = if base.episodes > 0 {
                format!(
                    " Baseline: {} episodes, success {:.0}%, {:.1} calls, {:.0} ms/task.",
                    base.episodes,
                    base.success * 100.0,
                    base.calls,
                    base.wall_ms
                )
            } else {
                " Baseline not measured yet: built on priors.".into()
            };
            (spec, agent_preds(&s), format!("{why}{src}"), lesson)
        }
    };
    spec.validate().ok()?;
    let est_ms = estimate_ms(&spec, cost, records);
    Some(Proposal {
        key,
        spec,
        predictions,
        rationale,
        lesson,
        label,
        est_ms,
    })
}

fn agent_spec(settings: AgentSettings) -> ExperimentSpec {
    ExperimentSpec::Agent {
        settings,
        baseline: settings,
        tasks: all_tasks(),
    }
}

fn agent_preds(s: &AgentStats) -> Vec<Prediction> {
    vec![
        pred("success", vec![s.success]),
        pred("calls", vec![s.calls]),
        pred("wall_ms", vec![s.wall_ms]),
    ]
}

fn pred(metric: &str, values: Vec<f64>) -> Prediction {
    Prediction {
        metric: metric.to_string(),
        values,
    }
}

/// Expected wall time of a spec from the cost model (and agent history).
pub fn estimate_ms(spec: &ExperimentSpec, cost: &CostModel, records: &[Record]) -> f64 {
    match spec {
        ExperimentSpec::Run {
            repeat,
            warmup,
            max_tokens,
        } => (*repeat + *warmup) as f64 * cost.call_ms(cost.base_prompt_tokens, *max_tokens as f64),
        ExperimentSpec::Sweep {
            var,
            values,
            repeat,
            warmup,
            max_tokens,
            ..
        } => {
            let per_point: Vec<f64> = values
                .iter()
                .map(|v| match var {
                    SweepVar::MaxTokens => {
                        cost.call_ms(cost.base_prompt_tokens, v.parse().unwrap_or(64.0))
                    }
                    SweepVar::PromptTokens => {
                        cost.call_ms(v.parse().unwrap_or(64.0), *max_tokens as f64)
                    }
                    _ => cost.call_ms(cost.base_prompt_tokens, *max_tokens as f64),
                })
                .collect();
            per_point.iter().sum::<f64>() * *repeat as f64 + per_point[0] * *warmup as f64
        }
        ExperimentSpec::Agent {
            settings, tasks, ..
        } => {
            let s = AgentStats::from_records(records, settings, tasks.len())
                .or_else(|| {
                    AgentStats::from_records(records, &AgentSettings::BASELINE, tasks.len())
                })
                .unwrap_or_else(|| AgentStats::prior(cost));
            s.wall_ms
                * tasks.len() as f64
                * if settings.critic == CriticMode::Model {
                    1.3
                } else {
                    1.0
                }
        }
    }
}

/// Settings for one autopilot session.
#[derive(Debug, Clone)]
pub struct Options {
    pub budget_min: f64,
    pub max_steps: usize,
    pub focus: Option<Focus>,
    #[allow(dead_code)] // --require-approval lands in a later commit
    pub require_approval: bool,
    pub tolerance_pct: f64,
    /// e.g. "--backend llama-server --template chatml " — prefixed to every printed command.
    pub backend_flags: String,
}

/// Only rows from the backend we are about to drive: a mock run says nothing
/// about llama-server, and vice versa.
pub fn relevant(records: &[Record], backend_name: &str) -> Vec<Record> {
    records
        .iter()
        .filter(|r| r.backend == backend_name)
        .cloned()
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // Interrupted/Declined land with approval + SIGINT handling
pub enum Stop {
    Budget { next_ms: f64 },
    MaxSteps,
    NothingLeft,
    Interrupted,
    Declined,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Budget { next_ms } => write!(
                f,
                "budget: the next trial (~{}) would exceed what is left",
                fmt_ms(*next_ms)
            ),
            Stop::MaxSteps => write!(f, "--max-steps reached"),
            Stop::NothingLeft => write!(f, "no high-value single-variable experiment remains"),
            Stop::Interrupted => {
                write!(f, "interrupted (Ctrl-C) after finishing the current trial")
            }
            Stop::Declined => write!(f, "stopped at the approval prompt"),
        }
    }
}

pub fn fmt_ms(ms: f64) -> String {
    if ms >= 60_000.0 {
        format!("{:.1} min", ms / 60_000.0)
    } else {
        format!("{:.1} s", ms / 1000.0)
    }
}

/// Decide whether the session may continue with `p`.
pub fn gate(state: &AutopilotState, p: Option<&Proposal>) -> Option<Stop> {
    if state.steps >= state.max_steps {
        return Some(Stop::MaxSteps);
    }
    let p = p?;
    if p.est_ms > state.remaining_ms() {
        return Some(Stop::Budget { next_ms: p.est_ms });
    }
    None
}

pub fn print_header(records: &[Record], backend_name: &str, opts: &Options, dry: bool) {
    let cost = CostModel::from_records(records);
    println!(
        "═══ AUTOPILOT{} ═══",
        if dry {
            " (dry run — nothing will execute)"
        } else {
            ""
        }
    );
    println!("backend   : {backend_name}");
    println!(
        "notebook  : {} rows from this backend",
        records
            .iter()
            .filter(|r| r.kind != Kind::Prediction)
            .count()
    );
    println!(
        "budget    : {:.0} min, at most {} trials{}",
        opts.budget_min,
        opts.max_steps,
        opts.focus
            .map(|f| format!(", focus {f:?}").to_lowercase())
            .unwrap_or_default()
    );
    println!(
        "cost model: decode {:.1} ms/token, prefill {:.2} ms/prompt token, overhead {:.1} ms ({})",
        cost.decode_ms_per_tok,
        cost.prefill_ms_per_tok,
        cost.overhead_ms,
        if cost.measured {
            "measured"
        } else {
            "PRIORS — nothing measured yet"
        }
    );
    println!("bottleneck: {}", cost.bottleneck());
    if advisor::levers(records).contains(&Lever::QuantNeedsRestart) {
        println!(
            "skipped   : quantization — a load-time knob. Autopilot never restarts the server; restart it yourself, tag runs (--tag quant=...), then `machine compare`."
        );
    }
}

pub fn print_proposal(n: usize, p: &Proposal, opts: &Options) {
    println!(
        "\n#{n:<2} {}   lesson {}   est {}",
        p.key,
        p.lesson,
        fmt_ms(p.est_ms)
    );
    println!("    why : {}", p.rationale);
    println!("    $ {}", p.command(&opts.backend_flags));
}

/// Show the sequence of trials the autopilot would run. Pure: no inference,
/// no writes. Each trial is assumed to take its estimated time.
pub fn dry_run(all: &[Record], backend_name: &str, opts: &Options) -> Vec<Proposal> {
    let records = relevant(all, backend_name);
    print_header(&records, backend_name, opts, true);
    let mut state = AutopilotState::new(opts.budget_min, opts.max_steps, opts.focus);
    let mut out = Vec::new();
    let stop = loop {
        let p = propose_next(&state, &records);
        if let Some(stop) = gate(&state, p.as_ref()) {
            break stop;
        }
        let Some(p) = p else { break Stop::NothingLeft };
        print_proposal(state.steps + 1, &p, opts);
        state.tried.insert(p.key.clone());
        state.spent_ms += p.est_ms;
        state.steps += 1;
        out.push(p);
    };
    println!(
        "\n── would run {} trial(s), ~{} of the {:.0} min budget. Stops: {stop}.",
        out.len(),
        fmt_ms(state.spent_ms),
        opts.budget_min
    );
    if stop == Stop::NothingLeft || out.is_empty() {
        return out;
    }
    println!(
        "   (a real run re-plans after every trial from the new measurements, so later trials may differ)"
    );
    out
}

/// Run one spec through the same code paths the CLI uses. The lab's label,
/// lesson, tolerance and predictions are set exactly as the printed command
/// would set them, so the notebook rows are indistinguishable from a human
/// run except for the `source=autopilot` tag.
pub fn execute_spec(lab: &mut Lab, p: &Proposal, tolerance_pct: f64) -> Result<Outcome> {
    lab.label = Some(p.label.clone());
    lab.lesson = Some(p.lesson);
    lab.cli_predictions = p.predictions.clone();
    lab.no_predict = false;
    lab.tolerance_pct = tolerance_pct;
    lab.tags.insert("source".into(), "autopilot".into());
    let params = |max_tokens: u32, ignore_eos: bool| GenParams {
        max_tokens,
        temperature: 0.0,
        seed: Some(42),
        cache_prompt: false,
        stop: Vec::new(),
        ignore_eos,
    };
    match &p.spec {
        ExperimentSpec::Run {
            repeat,
            warmup,
            max_tokens,
        } => {
            lab.quiet = false;
            experiment::run(
                lab,
                experiment::RunOpts {
                    prompt: experiment::DEFAULT_PROMPT.into(),
                    params: params(*max_tokens, true),
                    repeat: *repeat,
                    warmup: *warmup,
                    pad_to: None,
                    show_text: false,
                },
            )
        }
        ExperimentSpec::Sweep {
            var,
            values,
            repeat,
            warmup,
            max_tokens,
            metric,
        } => {
            lab.quiet = true;
            experiment::sweep(
                lab,
                experiment::SweepOpts {
                    var: *var,
                    values: values.clone(),
                    metric: Some(metric.to_string()),
                    base: experiment::RunOpts {
                        prompt: experiment::DEFAULT_PROMPT.into(),
                        params: params(*max_tokens, true),
                        repeat: *repeat,
                        warmup: *warmup,
                        pad_to: None,
                        show_text: false,
                    },
                },
            )
        }
        ExperimentSpec::Agent {
            settings, tasks, ..
        } => {
            lab.quiet = false;
            let mut gp = params(32, false); // `machine agent` default: one line per reply
            gp.temperature = settings.temperature;
            gp.cache_prompt = settings.cache_prompt;
            agent::run(
                lab,
                agent::AgentOpts {
                    tasks: tasks.clone(),
                    repeat: 1,
                    max_steps: 6,
                    critic: settings.critic,
                    memory_window: settings.memory_window,
                    params: gp,
                    verbose: false,
                },
            )
        }
    }
}

/// Measured means shaped like the predictions, for confirm-replications and learnings.
pub fn measured_like(preds: &[Prediction], o: &Outcome) -> Vec<Prediction> {
    preds
        .iter()
        .filter_map(|p| {
            let values: Option<Vec<f64>> =
                o.means.iter().map(|m| m.get(&p.metric).copied()).collect();
            Some(Prediction {
                metric: p.metric.clone(),
                values: values?,
            })
        })
        .collect()
}

/// Rows of one group (warm-ups excluded) at one sweep value (None = all).
fn group_values(records: &[Record], group: &str, value: Option<&str>, metric: &str) -> Vec<f64> {
    records
        .iter()
        .filter(|r| r.group == group && !r.warmup && matches!(r.kind, Kind::Run | Kind::Agent))
        .filter(|r| value.is_none_or(|v| r.value.as_deref() == Some(v)))
        .filter_map(|r| r.metrics.get(metric).copied())
        .collect()
}

fn pct(a: f64, b: f64) -> String {
    if a == 0.0 {
        format!("{:+}", metrics::short(b - a))
    } else {
        format!("{:+.0}%", 100.0 * (b - a) / a)
    }
}

fn first_run_row<'a>(records: &'a [Record], group: &str) -> Option<&'a Record> {
    records
        .iter()
        .find(|r| r.group == group && matches!(r.kind, Kind::Run | Kind::Agent))
}

/// Turn one trial into a structured learning. Deterministic: the text is a
/// template over the measured numbers, never free generation.
pub fn interpret(
    p: &Proposal,
    o: &Outcome,
    before: &[Record],
    after: &[Record],
    state: &AutopilotState,
    opts: &Options,
    lab: &Lab,
) -> Learning {
    let mut prediction = std::collections::BTreeMap::new();
    let mut actual = std::collections::BTreeMap::new();
    let mut error_pct = std::collections::BTreeMap::new();
    for pr in &p.predictions {
        prediction.insert(pr.metric.clone(), learnings::value_or_list(&pr.values));
        let a: Vec<f64> = o
            .means
            .iter()
            .filter_map(|m| m.get(&pr.metric).copied())
            .collect();
        if a.len() == pr.values.len() {
            actual.insert(pr.metric.clone(), learnings::value_or_list(&a));
        }
        let worst = o
            .results
            .iter()
            .filter(|r| r.metric.split('@').next() == Some(pr.metric.as_str()))
            .map(|r| r.error_pct)
            .fold(0.0_f64, f64::max);
        error_pct.insert(pr.metric.clone(), (worst * 10.0).round() / 10.0);
    }
    let passed = !o.results.is_empty() && o.results.iter().all(|r| r.passed);

    let mut single_variable = true;
    let mut sentences: Vec<String> = Vec::new();
    match &p.spec {
        ExperimentSpec::Sweep {
            var,
            values,
            metric,
            ..
        } => {
            let (lo, hi) = (&values[0], &values[values.len() - 1]);
            let a = Summary::of(&group_values(after, &o.group, Some(lo), metric));
            let b = Summary::of(&group_values(after, &o.group, Some(hi), metric));
            if let (Some(a), Some(b)) = (a, b) {
                sentences.push(format!(
                    "Changing only {} from {lo} to {hi} moved {metric} from {} to {} ({}, {}).",
                    var.name(),
                    metrics::fmt(metric, a.mean),
                    metrics::fmt(metric, b.mean),
                    pct(a.mean, b.mean),
                    stats::verdict(stats::welch_t(&a, &b))
                ));
            }
            if *var == SweepVar::CachePrompt {
                let pt = |v: &str| {
                    Summary::of(&group_values(after, &o.group, Some(v), "prompt_tokens"))
                        .map(|s| s.mean)
                };
                if let (Some(cold), Some(warm)) = (pt("false"), pt("true")) {
                    sentences.push(format!(
                        "On a cache hit the server evaluated {warm:.0} of {cold:.0} prompt tokens."
                    ));
                }
            }
        }
        ExperimentSpec::Run { .. } => {
            let d = Summary::of(&group_values(after, &o.group, None, "decode_tok_s"));
            let w = Summary::of(&group_values(after, &o.group, None, "wall_ms"));
            if let (Some(d), Some(w)) = (d, w) {
                sentences.push(format!(
                    "Over {} runs decode was {:.1} tok/s (cv {:.1}%) and wall {}; the smallest trustworthy effect here is ~{:.0}%.",
                    d.n,
                    d.mean,
                    d.cv() * 100.0,
                    metrics::fmt("wall_ms", w.mean),
                    (2.0 * d.cv() * 100.0).max(1.0)
                ));
            }
        }
        ExperimentSpec::Agent {
            settings,
            baseline,
            tasks,
        } => {
            let m = &o.means[0];
            let get = |k: &str| m.get(k).copied().unwrap_or(0.0);
            let base = if settings != baseline {
                AgentStats::from_records(before, baseline, tasks.len())
            } else {
                None
            };
            match base {
                Some(b) => {
                    // Reuse compare's multi-setting detector against the baseline rows.
                    let base_row = before.iter().rev().find(|r| baseline.matches(r));
                    if let (Some(x), Some(y)) = (base_row, first_run_row(after, &o.group)) {
                        let diff = experiment::differing_settings(x, y);
                        single_variable = experiment::single_variable(&diff);
                        if !single_variable {
                            sentences.push(format!(
                                "Confounded: differs from the baseline in {diff:?}, so it is not used for conclusions."
                            ));
                        }
                    }
                    sentences.push(format!(
                        "Changing only {} moved success {:.0}% → {:.0}%, calls {:.1} → {:.1}, and wall {} → {} per task ({}).",
                        settings.diff(baseline).join(", "),
                        b.success * 100.0,
                        get("success") * 100.0,
                        b.calls,
                        get("calls"),
                        metrics::fmt("wall_ms", b.wall_ms),
                        metrics::fmt("wall_ms", get("wall_ms")),
                        pct(b.wall_ms, get("wall_ms"))
                    ));
                }
                None => sentences.push(format!(
                    "Agent at these settings: success {:.0}%, {:.1} calls and {} per task, {:.0} prompt tokens per call.",
                    get("success") * 100.0,
                    get("calls"),
                    metrics::fmt("wall_ms", get("wall_ms")),
                    get("prompt_tokens") / get("calls").max(1.0)
                )),
            }
        }
    }
    let worst = o
        .results
        .iter()
        .max_by(|a, b| a.error_pct.total_cmp(&b.error_pct));
    sentences.push(match worst {
        None => "No prediction was scored.".into(),
        Some(_) if passed => format!(
            "All {} prediction(s) were within ±{:.0}%, so the cost model explains this result.",
            o.results.len(),
            opts.tolerance_pct
        ),
        Some(w) => format!(
            "The {} prediction missed by {:.0}%, so the assumption behind it does not hold here.",
            w.metric, w.error_pct
        ),
    });
    sentences.truncate(3);

    // What the numbers now suggest: the proposer's next pick on the updated notebook.
    // The state already lists this trial as tried (and as missed, if it missed).
    let next_priority = match propose_next(state, after) {
        Some(n) => format!(
            "{}: {}",
            n.key,
            n.rationale
                .split(". ")
                .next()
                .unwrap_or(&n.rationale)
                .trim_end_matches('.')
        ),
        None => format!("nothing left in scope. Advisor: {}", advisor::next(after).0),
    };

    Learning {
        kind: "learning".into(),
        ts: learnings::iso8601(now_secs()),
        experiment: p.command(&opts.backend_flags),
        prediction,
        actual,
        error_pct,
        passed,
        learning: sentences.join(" "),
        next_priority,
        group: o.group.clone(),
        key: p.key.clone(),
        single_variable,
        backend: lab.backend.name(),
        simulated: lab.simulated(),
    }
}

/// What happened in one autopilot session.
#[derive(Debug)]
pub struct Session {
    pub trials: Vec<(Proposal, Outcome)>,
    pub stop: Stop,
    pub spent_ms: f64,
}

/// The real loop: observe → propose → (approve) → run → score → learn → repeat.
pub fn execute(lab: &mut Lab, opts: &Options) -> Result<Session> {
    let backend_name = lab.backend.name();
    let mut state = AutopilotState::new(opts.budget_min, opts.max_steps, opts.focus);
    let start = std::time::Instant::now();
    let all = lab.notebook.all()?;
    print_header(&relevant(&all, &backend_name), &backend_name, opts, false);
    let mut trials = Vec::new();
    let stop = loop {
        let records = relevant(&lab.notebook.all()?, &backend_name);
        state.bottleneck = CostModel::from_records(&records).bottleneck();
        state.spent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let p = propose_next(&state, &records);
        if let Some(stop) = gate(&state, p.as_ref()) {
            break stop;
        }
        let Some(p) = p else { break Stop::NothingLeft };
        print_proposal(state.steps + 1, &p, opts);
        println!();
        let outcome = execute_spec(lab, &p, opts.tolerance_pct)?;
        state.steps += 1;
        state.tried.insert(p.key.clone());
        if outcome.results.iter().any(|r| !r.passed) {
            state.missed.push((
                p.key.clone(),
                p.spec.clone(),
                measured_like(&p.predictions, &outcome),
            ));
        }
        let after = relevant(&lab.notebook.all()?, &backend_name);
        let l = interpret(&p, &outcome, &records, &after, &state, opts, lab);
        learnings::append(&lab.notebook, &l)?;
        println!(
            "\n── LEARNING ({})",
            if l.passed {
                "predictions held"
            } else {
                "prediction missed"
            }
        );
        println!("  {}", l.learning);
        println!("  next: {}", l.next_priority);
        state.remember(l.learning.clone());
        trials.push((p, outcome));
    };
    let spent_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("\n═══ AUTOPILOT DONE ═══");
    Ok(Session {
        trials,
        stop,
        spent_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_notebook_proposes_baseline_then_sweeps_then_agents() {
        let mut st = AutopilotState::new(1e9, 100, None);
        let mut keys = Vec::new();
        while let Some(p) = propose_next(&st, &[]) {
            p.spec.validate().unwrap();
            assert!(!p.predictions.is_empty(), "{} has no predictions", p.key);
            st.tried.insert(p.key.clone());
            keys.push(p.key);
            assert!(keys.len() < 50, "proposer does not terminate");
        }
        assert_eq!(keys[0], "baseline");
        assert!(
            !keys.iter().any(|k| k.starts_with("noise#")),
            "baseline already measures noise"
        );
        assert!(keys.contains(&"sweep:cache_prompt".to_string()));
        let pos = |k: &str| keys.iter().position(|x| x == k).unwrap();
        assert!(pos("agent:baseline") < pos("agent:critic=model"));
        assert!(pos("agent:baseline") < pos("agent:cache_prompt"));
    }

    #[test]
    fn deterministic() {
        let st = AutopilotState::new(30.0, 40, None);
        let a = propose_next(&st, &[]).unwrap();
        let b = propose_next(&st, &[]).unwrap();
        assert_eq!(a.spec, b.spec);
        assert_eq!(a.command(""), b.command(""));
    }

    #[test]
    fn focus_filters_but_keeps_prerequisites() {
        let mut st = AutopilotState::new(1e9, 100, Some(Focus::Cache));
        let mut keys = Vec::new();
        while let Some(p) = propose_next(&st, &[]) {
            st.tried.insert(p.key.clone());
            keys.push(p.key);
        }
        assert!(keys.contains(&"baseline".to_string()));
        assert!(keys.contains(&"sweep:cache_prompt".to_string()));
        assert!(keys.contains(&"agent:cache_prompt".to_string()));
        assert!(
            !keys
                .iter()
                .any(|k| k == "sweep:max_tokens" || k.starts_with("agent:critic"))
        );
    }

    #[test]
    fn bare_agent_command_keeps_subcommand() {
        let p = Proposal {
            key: "agent:baseline".into(),
            spec: agent_spec(AgentSettings::BASELINE),
            predictions: vec![pred("calls", vec![2.0])],
            rationale: String::new(),
            lesson: 4,
            label: "x".into(),
            est_ms: 0.0,
        };
        assert_eq!(
            p.command("--backend mock "),
            "machine agent --backend mock --lesson 4 --label x --predict calls=2.0"
        );
    }

    #[test]
    fn rejects_multi_variable_agent_spec() {
        let mut s = AgentSettings::BASELINE;
        s.critic = CriticMode::Model;
        s.cache_prompt = true;
        let spec = ExperimentSpec::Agent {
            settings: s,
            baseline: AgentSettings::BASELINE,
            tasks: all_tasks(),
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn commands_are_typeable() {
        let st = AutopilotState::new(30.0, 40, Some(Focus::Cache));
        let mut st2 = st.clone();
        st2.tried.insert("baseline".into());
        let p = propose_next(&st2, &[]).unwrap();
        let cmd = p.command("--backend mock ");
        assert!(
            cmd.starts_with("machine sweep --backend mock --var cache-prompt"),
            "{cmd}"
        );
        assert!(cmd.contains("--predict prefill_ms="));
    }
}
