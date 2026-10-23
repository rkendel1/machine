//! Experiment selection by expected learning and practical value.
//!
//! For each candidate (always one validated single-variable spec):
//!
//!   gain  = max over target beliefs of  u · (1 − √(n / (n + m)))
//!           u = current relative uncertainty in [0,1] (1 if untested),
//!           n = observations behind the belief, m = observations the trial adds.
//!           (A 95% interval shrinks like 1/√n; untested beliefs score u = 1, factor 1.)
//!   value = how much the target governs agent wall time or success, in [0,1],
//!           read off the surrogate (e.g. prefill's share of an agent call).
//!   score = gain · (0.25 + 0.75 · value) / (1 + log2(1 + cost_s / 10))
//!
//! The weights are judgment calls and are printed with every ranking so they
//! can be argued with. Candidates below `MIN_SCORE` are "low value": the
//! loop stops rather than run them.

use super::BASELINE_KEY;
use super::analyze::{self, Analysis};
use super::beliefs::{BeliefState, Status};
use crate::agent::CriticMode;
use crate::autopilot::{
    self, AgentSettings, AutopilotState, CostModel, ExperimentSpec, Focus, Planner, Proposal,
};
use crate::experiment::SweepVar;
use crate::notebook::{Kind, Record};
use crate::predict::Prediction;
use std::collections::BTreeSet;

/// Below this score an experiment is not worth its time.
pub const MIN_SCORE: f64 = 0.02;
/// With fewer run rows than this, beliefs are too thin to steer by: use the checklist.
pub const MIN_ROWS: usize = 15;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub proposal: Proposal,
    pub targets: Vec<String>,
    pub gain: f64,
    pub value: f64,
    pub cost_ms: f64,
    pub score: f64,
    pub why: String,
    pub source: &'static str,
}

fn pred(metric: &str, values: Vec<f64>) -> Prediction {
    Prediction {
        metric: metric.into(),
        values,
    }
}

/// Measured target→evaluated token ratio of padded prompts (the tokenizer
/// decides the real length). 1.0 until a prompt-tokens sweep exists.
pub fn pad_ratio(records: &[Record]) -> f64 {
    let mut r: Vec<f64> = records
        .iter()
        .filter(|r| r.kind == Kind::Run && !r.warmup && r.experiment == "sweep:prompt_tokens")
        .filter_map(|r| {
            let target: f64 = r.value.as_ref()?.parse().ok()?;
            Some(r.metrics.get("prompt_tokens")? / target)
        })
        .collect();
    super::fit::median(&mut r).unwrap_or(1.0)
}

fn value_of(key: &str, a: &Analysis) -> f64 {
    let b = &a.beliefs;
    let share = a.surrogate.agent_prefill_share().unwrap_or(0.5);
    let base = b.agent_configs.iter().find(|c| c.key == BASELINE_KEY);
    let plausible = |k: &str| -> Option<f64> {
        let e = b.get(k)?.est?;
        Some(e.value.abs().max(e.lo.abs()).max(e.hi.abs()))
    };
    match key {
        "prefill.ms_per_token" | "effect.cache_hit.prefill_ms" | "interaction.cache_x_prompt" => {
            share
        }
        "prefill.fixed_ms" => 0.5 * share,
        "decode.ms_per_token" => 1.0 - share,
        "decode.fixed_ms" | "overhead.ms" => 0.1,
        "noise.decode_cv" => 0.3,
        "effect.temperature.decode_tok_s" => 0.1,
        "agent.baseline.success" => 1.0,
        "agent.baseline.wall_ms" => 0.8,
        k if k.starts_with("effect.agent.") && k.ends_with(".success") => {
            plausible(k).map(|v| v.min(1.0)).unwrap_or(0.5)
        }
        k if k.starts_with("effect.agent.") && k.ends_with(".wall_ms") => {
            match (plausible(k), base) {
                (Some(v), Some(c)) if c.wall_ms.value > 0.0 => (v / c.wall_ms.value).min(1.0),
                _ if k.contains("cache_prompt") => share,
                _ => 0.3,
            }
        }
        k if k.starts_with("effect.agent.") && k.ends_with(".calls") => {
            match (plausible(k), base) {
                (Some(v), Some(c)) if c.calls.value > 0.0 => (v / c.calls.value).min(1.0),
                _ => 0.3,
            }
        }
        _ => 0.0,
    }
}

fn gain_of(key: &str, added: usize, b: &BeliefState) -> f64 {
    match b.get(key) {
        None => 0.0,
        Some(x) if x.status == Status::Untested || x.est.is_none() => 1.0,
        Some(x) => {
            let n = x.n.max(1) as f64;
            x.uncertainty() * (1.0 - (n / (n + added as f64)).sqrt())
        }
    }
}

fn added(spec: &ExperimentSpec) -> usize {
    match spec {
        ExperimentSpec::Run { repeat, .. } => *repeat,
        ExperimentSpec::Sweep { values, repeat, .. } => values.len() * repeat,
        ExperimentSpec::Agent { tasks, .. } => tasks.len(),
    }
}

fn targets_for_key(key: &str) -> Vec<String> {
    let k = key.strip_prefix("confirm:").unwrap_or(key);
    let v: &[&str] = match k {
        "baseline" => &["decode.ms_per_token", "prefill.ms_per_token"],
        k if k.starts_with("noise#") => &["noise.decode_cv"],
        "sweep:max_tokens" | "refine:decode" => &["decode.ms_per_token", "decode.fixed_ms"],
        "sweep:prompt_tokens" | "refine:prefill_fixed" => {
            &["prefill.fixed_ms", "prefill.ms_per_token"]
        }
        "sweep:temperature" => &["effect.temperature.decode_tok_s"],
        "sweep:cache_prompt" => &["effect.cache_hit.prefill_ms", "prefill.fixed_ms"],
        "interaction:cache_x_prompt" => &["interaction.cache_x_prompt"],
        "agent:baseline" | "replicate:agent:baseline" => &[
            "agent.baseline.success",
            "agent.baseline.wall_ms",
            "effect.agent.critic=model.success",
            "effect.agent.memory_window=1.success",
            "effect.agent.cache_prompt.wall_ms",
        ],
        _ => &[],
    };
    if !v.is_empty() {
        return v.iter().map(|s| s.to_string()).collect();
    }
    // agent:<knob> and replicate:agent:<knob>
    let knob = k
        .trim_start_matches("replicate:")
        .trim_start_matches("agent:");
    ["success", "calls", "wall_ms"]
        .iter()
        .map(|m| format!("effect.agent.{knob}.{m}"))
        .collect()
}

fn knob_settings(knob: &str) -> Option<AgentSettings> {
    let b = AgentSettings::BASELINE;
    Some(match knob {
        "critic=model" => AgentSettings {
            critic: CriticMode::Model,
            ..b
        },
        "critic=oracle" => AgentSettings {
            critic: CriticMode::Oracle,
            ..b
        },
        "memory_window=1" => AgentSettings {
            memory_window: Some(1),
            ..b
        },
        "cache_prompt" => AgentSettings {
            cache_prompt: true,
            ..b
        },
        "temperature=0.8" => AgentSettings {
            temperature: 0.8,
            ..b
        },
        "baseline" => b,
        _ => return None,
    })
}

fn focus_ok(key: &str, focus: Option<Focus>) -> bool {
    let Some(f) = focus else { return true };
    let k = key.strip_prefix("confirm:").unwrap_or(key);
    match k {
        "refine:prefill_fixed" => f == Focus::Prefill,
        "interaction:cache_x_prompt" => matches!(f, Focus::Cache | Focus::Prefill),
        "refine:decode" => f == Focus::Decode,
        k if k.starts_with("replicate:agent:critic") => f == Focus::Critic,
        k if k.starts_with("replicate:agent:cache") => matches!(f, Focus::Cache | Focus::Agent),
        k if k.starts_with("replicate:agent:") => matches!(f, Focus::Agent | Focus::Critic),
        _ => true, // lever candidates were already focus-filtered by the checklist
    }
}

/// Surrogate-based predictions for run/sweep specs; agent specs keep theirs.
fn surrogate_predictions(
    spec: &ExperimentSpec,
    a: &Analysis,
    ratio: f64,
) -> Option<Vec<Prediction>> {
    let s = &a.surrogate;
    let base = s.base_prompt_tokens;
    let wall = |p: f64, n: f64, hit: bool| {
        s.call(&super::surrogate::CallQuery {
            prompt_tokens: p,
            max_tokens: n,
            cache_hit: hit,
        })
        .map(|c| c.wall_ms.value)
    };
    let tok_s = |n: f64| {
        let d = s.decode.as_ref()?;
        let ms = d.at(n);
        (ms > 0.0).then(|| n * 1000.0 / ms)
    };
    match spec {
        ExperimentSpec::Run { max_tokens, .. } => {
            let n = *max_tokens as f64;
            Some(vec![
                pred("decode_tok_s", vec![tok_s(n)?]),
                pred("wall_ms", vec![wall(base, n, false)?]),
            ])
        }
        ExperimentSpec::Sweep {
            var,
            values,
            max_tokens,
            metric,
            pad_to,
            ..
        } => {
            let n = *max_tokens as f64;
            let prompt = pad_to.map(|p| p as f64 * ratio).unwrap_or(base);
            let vals: Option<Vec<f64>> = values
                .iter()
                .map(|v| match (var, *metric) {
                    (SweepVar::MaxTokens, "wall_ms") => wall(prompt, v.parse().ok()?, false),
                    (SweepVar::PromptTokens, "prefill_ms") => {
                        Some(s.prefill_at(v.parse::<f64>().ok()? * ratio)?.value.max(0.0))
                    }
                    (SweepVar::CachePrompt, "prefill_ms") => {
                        let p = if v == "true" { 1.0 } else { prompt };
                        Some(s.prefill_at(p)?.value.max(0.0))
                    }
                    (SweepVar::Temperature, "decode_tok_s") => tok_s(n),
                    _ => None,
                })
                .collect();
            Some(vec![pred(metric, vals?)])
        }
        ExperimentSpec::Agent { .. } => None,
    }
}

fn agent_replication(knob: &str, a: &Analysis) -> Option<(ExperimentSpec, Vec<Prediction>)> {
    let s = knob_settings(knob)?;
    let c = a.surrogate.agent_config(&analyze::config_key(&s))?;
    let spec = ExperimentSpec::Agent {
        settings: s,
        baseline: s,
        tasks: autopilot::all_tasks(),
    };
    Some((
        spec,
        vec![
            pred("success", vec![c.success.value]),
            pred("calls", vec![c.calls.value]),
            pred("wall_ms", vec![c.wall_ms.value]),
        ],
    ))
}

fn scientist_specs(
    a: &Analysis,
    ratio: f64,
) -> Vec<(String, ExperimentSpec, Vec<Prediction>, String, u8)> {
    let b = &a.beliefs;
    let mut out = Vec::new();
    let measured = |k: &str| b.get(k).is_some_and(|x| x.est.is_some());
    let sweep = |var, values: &[&str], repeat, max_tokens, metric, pad_to| ExperimentSpec::Sweep {
        var,
        values: values.iter().map(|s| s.to_string()).collect(),
        repeat,
        warmup: 1,
        max_tokens,
        metric,
        pad_to,
    };
    if measured("prefill.ms_per_token") {
        let spec = sweep(
            SweepVar::PromptTokens,
            &["32", "64", "128"],
            3,
            1,
            "prefill_ms",
            None,
        );
        if let Some(p) = surrogate_predictions(&spec, a, ratio) {
            out.push((
                "refine:prefill_fixed".into(),
                spec,
                p,
                "Short prompts pin down the fixed prefill cost (the intercept), which bounds every cached call.".into(),
                3,
            ));
        }
    }
    if measured("effect.cache_hit.prefill_ms") {
        let spec = sweep(
            SweepVar::CachePrompt,
            &["false", "true"],
            3,
            8,
            "prefill_ms",
            Some(512),
        );
        if let Some(p) = surrogate_predictions(&spec, a, ratio) {
            out.push((
                "interaction:cache_x_prompt".into(),
                spec,
                p,
                "Cache benefit was only measured at the default prompt. Holding the prompt at ~512 tokens tests whether a hit stays at the fixed cost (null: no interaction).".into(),
                3,
            ));
        }
    }
    if measured("decode.ms_per_token") {
        let spec = sweep(
            SweepVar::MaxTokens,
            &["16", "64", "128"],
            3,
            64,
            "wall_ms",
            None,
        );
        if let Some(p) = surrogate_predictions(&spec, a, ratio) {
            out.push((
                "refine:decode".into(),
                spec,
                p,
                "More output lengths tighten the decode line.".into(),
                3,
            ));
        }
    }
    let mut knobs: BTreeSet<&str> = BTreeSet::new();
    for x in b
        .beliefs
        .iter()
        .filter(|x| x.key.starts_with("effect.agent.") && x.est.is_some())
    {
        if x.status == Status::Uncertain {
            let knob = x
                .key
                .trim_start_matches("effect.agent.")
                .rsplit_once('.')
                .map(|k| k.0);
            if let Some(k) = knob {
                knobs.insert(k);
            }
        }
    }
    for knob in knobs {
        if let Some((spec, p)) = agent_replication(knob, a) {
            out.push((
                format!("replicate:agent:{knob}"),
                spec,
                p,
                format!("The effect of {knob} on the agent is measured but its interval still includes zero; another 10 episodes narrow it."),
                6,
            ));
        }
    }
    if b.get("agent.baseline.success")
        .is_some_and(|x| x.est.is_some())
        && let Some((spec, p)) = agent_replication("baseline", a)
    {
        out.push((
            "replicate:agent:baseline".into(),
            spec,
            p,
            "Every agent effect is measured against the baseline; its own 10 episodes are the shared bottleneck in those intervals.".into(),
            4,
        ));
    }
    out
}

/// All candidates, scored and sorted (best first). Pure.
pub fn rank(a: &Analysis, records: &[Record], state: &AutopilotState) -> Vec<Candidate> {
    let cost = CostModel::from_records(records);
    let ratio = pad_ratio(records);
    let mut raw: Vec<(Proposal, &'static str, String)> = Vec::new();

    // Lever candidates: everything the checklist would still propose, in focus.
    let mut st = state.clone();
    for _ in 0..64 {
        let Some(mut p) = autopilot::propose_next(&st, records) else {
            break;
        };
        st.tried.insert(p.key.clone());
        if let Some(sp) = surrogate_predictions(&p.spec, a, ratio) {
            p.predictions = sp;
        }
        let why = p
            .rationale
            .split(". ")
            .next()
            .unwrap_or("")
            .trim_end_matches('.')
            .to_string();
        raw.push((p, "lever", why));
    }
    // Scientist candidates: refinements, interactions, replications.
    for (key, spec, predictions, why, lesson) in scientist_specs(a, ratio) {
        if state.tried.contains(&key) || !focus_ok(&key, state.focus) || spec.validate().is_err() {
            continue;
        }
        let est_ms = autopilot::estimate_ms(&spec, &cost, records);
        let label = format!("sci-{}", key.replace([':', '=', '#'], "-"));
        raw.push((
            Proposal {
                key,
                spec,
                predictions,
                rationale: why.clone(),
                lesson,
                label,
                est_ms,
            },
            if why.starts_with("Cache benefit") {
                "interaction"
            } else {
                "refine"
            },
            why,
        ));
    }

    let mut out: Vec<Candidate> = raw
        .into_iter()
        .map(|(proposal, source, why)| {
            let replicate = proposal.key.starts_with("replicate:");
            // A replication can only tighten what is already measured; it
            // creates no new belief, so untested targets don't count for it.
            let targets: Vec<String> = targets_for_key(&proposal.key)
                .into_iter()
                .filter(|t| !replicate || a.beliefs.get(t).is_some_and(|b| b.est.is_some()))
                .collect();
            let m = added(&proposal.spec);
            let (mut gain, mut value) = (0.0_f64, 0.0_f64);
            for t in &targets {
                let (g, v) = (gain_of(t, m, &a.beliefs), value_of(t, a));
                if g > gain + 1e-12 || ((g - gain).abs() <= 1e-12 && v > value) {
                    gain = g;
                    value = v;
                }
            }
            let cost_ms = proposal.est_ms.max(1.0);
            let score = gain * (0.25 + 0.75 * value) / (1.0 + (1.0 + cost_ms / 10_000.0).log2());
            let source = if proposal.key.starts_with("replicate:") {
                "replicate"
            } else {
                source
            };
            Candidate {
                proposal,
                targets,
                gain,
                value,
                cost_ms,
                score,
                why,
                source,
            }
        })
        .collect();
    out.sort_by(|x, y| {
        y.score
            .total_cmp(&x.score)
            .then_with(|| x.proposal.key.cmp(&y.proposal.key))
    });
    out
}

/// The selector as a planner for the shared autopilot loop.
pub struct Scientist {
    /// Beliefs before the latest trial, to report what changed.
    pub last: Option<BeliefState>,
    /// Where to write beliefs.json / beliefs_history.jsonl (None in dry runs).
    pub save_dir: Option<std::path::PathBuf>,
    pub backend: String,
}

impl Scientist {
    pub fn new(backend: &str, save_dir: Option<std::path::PathBuf>) -> Self {
        Self {
            last: None,
            save_dir,
            backend: backend.into(),
        }
    }
}

/// Is the data too thin for beliefs to steer? Then the checklist decides.
pub fn sparse(a: &Analysis) -> bool {
    a.beliefs.run_rows < MIN_ROWS || a.surrogate.prefill.is_none() || a.surrogate.decode.is_none()
}

impl Planner for Scientist {
    fn name(&self) -> &'static str {
        "scientist (expected information gain × value ÷ cost)"
    }

    fn propose(&mut self, state: &AutopilotState, records: &[Record]) -> Option<Proposal> {
        let a = analyze::analyze(records, &[], &self.backend);
        if self.last.is_none() {
            self.last = Some(a.beliefs.clone());
        }
        if sparse(&a) {
            let mut p = autopilot::propose_next(state, records)?;
            p.rationale = format!(
                "[fallback: sparse data, deterministic checklist] {}",
                p.rationale
            );
            return Some(p);
        }
        let best = rank(&a, records, state).into_iter().next()?;
        if best.score < MIN_SCORE {
            return None;
        }
        let mut p = best.proposal;
        p.rationale = format!(
            "[score {:.3} = gain {:.2} × value {:.2} / cost {}] {}",
            best.score,
            best.gain,
            best.value,
            autopilot::fmt_ms(best.cost_ms),
            p.rationale
        );
        Some(p)
    }

    fn after_trial(&mut self, records: &[Record]) {
        let a = analyze::analyze(records, &[], &self.backend);
        if let Some(prev) = &self.last {
            let changes = belief_changes(prev, &a.beliefs);
            if !changes.is_empty() {
                println!("\n── BELIEF UPDATE");
                for c in changes {
                    println!("  {c}");
                }
            }
        }
        if let Some(dir) = &self.save_dir {
            let _ = a.beliefs.save(dir);
            let _ = append_history(dir, &a.beliefs);
        }
        self.last = Some(a.beliefs);
    }
}

/// Human-readable diffs between two belief states (status changes and
/// uncertainty that moved by ≥10% of its value).
pub fn belief_changes(prev: &BeliefState, now: &BeliefState) -> Vec<String> {
    let mut out = Vec::new();
    for b in &now.beliefs {
        let Some(p) = prev.get(&b.key) else { continue };
        let (u0, u1) = (p.uncertainty(), b.uncertainty());
        if p.status != b.status || (u0 - u1).abs() >= 0.1 * u0.max(1e-9) && (p.est != b.est) {
            out.push(format!(
                "{:<38} {} → {}   {} → {}",
                b.key,
                p.fmt_est(),
                b.fmt_est(),
                p.status.tag().trim(),
                b.status.tag().trim()
            ));
        }
    }
    out
}

fn append_history(dir: &std::path::Path, b: &BeliefState) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("beliefs_history.jsonl"))?;
    let compact: Vec<serde_json::Value> = b
        .beliefs
        .iter()
        .map(|x| serde_json::json!({"key": x.key, "est": x.est, "n": x.n, "status": x.status}))
        .collect();
    writeln!(
        f,
        "{}",
        serde_json::json!({"ts": b.fitted_at, "beliefs": compact})
    )?;
    Ok(())
}

/// Print a ranking with every score component.
pub fn print_ranking(cands: &[Candidate], top: usize, flags: &str) {
    println!(
        "  {:<3} {:<30} {:>6} {:>5} {:>5} {:>8}  {:<11} targets",
        "#", "candidate", "score", "gain", "value", "cost", "kind"
    );
    for (i, c) in cands.iter().take(top).enumerate() {
        println!(
            "  {:<3} {:<30} {:>6.3} {:>5.2} {:>5.2} {:>8}  {:<11} {}",
            i + 1,
            c.proposal.key,
            c.score,
            c.gain,
            c.value,
            autopilot::fmt_ms(c.cost_ms),
            c.source,
            c.targets.join(", ")
        );
    }
    if let Some(c) = cands.first() {
        println!("\n  top pick: {}", c.why);
        println!("  $ {}", c.proposal.command(flags));
    }
    let low: Vec<&str> = cands
        .iter()
        .filter(|c| c.score < MIN_SCORE)
        .map(|c| c.proposal.key.as_str())
        .collect();
    if !low.is_empty() {
        println!(
            "\n  low value (score < {MIN_SCORE}, won't run): {}",
            low.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untested_targets_have_full_gain() {
        let a = analyze::analyze(&[], &[], "mock (SIMULATED)");
        assert_eq!(gain_of("interaction.cache_x_prompt", 6, &a.beliefs), 1.0);
        assert_eq!(gain_of("no.such.belief", 6, &a.beliefs), 0.0);
        assert!(sparse(&a));
    }

    #[test]
    fn every_key_maps_to_known_beliefs() {
        let a = analyze::analyze(&[], &[], "mock (SIMULATED)");
        for k in [
            "baseline",
            "noise#1",
            "sweep:max_tokens",
            "sweep:prompt_tokens",
            "sweep:temperature",
            "sweep:cache_prompt",
            "agent:baseline",
            "agent:critic=model",
            "agent:critic=oracle",
            "agent:memory_window=1",
            "agent:cache_prompt",
            "refine:prefill_fixed",
            "interaction:cache_x_prompt",
            "refine:decode",
            "replicate:agent:memory_window=1",
            "confirm:sweep:cache_prompt",
        ] {
            let ts = targets_for_key(k);
            assert!(!ts.is_empty(), "{k}");
            for t in ts {
                assert!(a.beliefs.get(&t).is_some(), "{k} → {t} is not a belief");
            }
        }
    }
}
