//! Progress gating and "what next" — decisions made from recorded numbers only.

use crate::lessons::{self, LESSONS};
use crate::notebook::{Kind, Record};
use crate::stats::Summary;
use std::collections::BTreeSet;

pub struct LessonProgress {
    pub n: u8,
    pub attempts: usize,
    pub passes: usize,
    pub last_error_pct: Option<f64>,
    pub mean_error_pct: Option<f64>,
    pub passed: bool,
    /// Autopilot predictions for this lesson (shown, never used for gating).
    pub auto_attempts: usize,
    pub auto_passes: usize,
}

fn belongs(r: &Record, n: u8) -> bool {
    if r.lesson == Some(n) {
        return true;
    }
    if r.lesson.is_some() {
        return false;
    }
    lessons::get(n).is_some_and(|l| l.counts.iter().any(|p| r.experiment.starts_with(p)))
}

/// Rows written by `machine autopilot` carry `source=autopilot`.
pub fn is_autopilot(r: &Record) -> bool {
    r.tags.get("source").map(String::as_str) == Some("autopilot")
}

/// A lesson is passed when at least 2 of *your* predictions were within
/// tolerance and the most recent one was too — "can you predict the next
/// change reliably?" Autopilot predictions are machine predictions: they are
/// reported alongside but never pass a lesson for you.
pub fn progress(records: &[Record]) -> Vec<LessonProgress> {
    LESSONS
        .iter()
        .map(|l| {
            let preds = |auto: bool| -> Vec<&crate::notebook::PredictionResult> {
                records
                    .iter()
                    .filter(|r| {
                        r.kind == Kind::Prediction && belongs(r, l.n) && is_autopilot(r) == auto
                    })
                    .flat_map(|r| r.predictions.iter())
                    .collect()
            };
            let human = preds(false);
            let auto = preds(true);
            let passes = human.iter().filter(|p| p.passed).count();
            let errs: Vec<f64> = human.iter().map(|p| p.error_pct).collect();
            LessonProgress {
                n: l.n,
                attempts: human.len(),
                passes,
                last_error_pct: human.last().map(|p| p.error_pct),
                mean_error_pct: Summary::of(&errs).map(|s| s.mean),
                passed: passes >= 2 && human.last().is_some_and(|p| p.passed),
                auto_attempts: auto.len(),
                auto_passes: auto.iter().filter(|p| p.passed).count(),
            }
        })
        .collect()
}

pub fn print_progress(records: &[Record]) {
    println!(
        "═══ PROGRESS ═══  (a lesson passes after ≥2 of YOUR predictions within tolerance, the latest included)\n"
    );
    let ps = progress(records);
    let mut unlocked = true;
    for p in &ps {
        let l = lessons::get(p.n).unwrap();
        let status = if p.passed {
            "PASSED "
        } else if unlocked {
            unlocked = false;
            "CURRENT"
        } else {
            "locked "
        };
        let auto = if p.auto_attempts > 0 {
            format!("  autopilot {}/{}", p.auto_passes, p.auto_attempts)
        } else {
            String::new()
        };
        println!(
            "  [{status}] {}. {:<30} predictions {:>2}  within tol {:>2}  mean err {:>6}  last err {:>6}{auto}",
            p.n,
            l.title,
            p.attempts,
            p.passes,
            p.mean_error_pct
                .map(|e| format!("{e:.0}%"))
                .unwrap_or("—".into()),
            p.last_error_pct
                .map(|e| format!("{e:.0}%"))
                .unwrap_or("—".into()),
        );
    }
    let sim = records
        .iter()
        .filter(|r| r.simulated && r.kind != Kind::Prediction)
        .count();
    let real = records
        .iter()
        .filter(|r| !r.simulated && r.kind != Kind::Prediction)
        .count();
    let auto = records
        .iter()
        .filter(|r| r.kind != Kind::Prediction && is_autopilot(r))
        .count();
    println!("\n  rows in notebook: {real} real, {sim} simulated ({auto} written by autopilot)");
    if ps.iter().any(|p| p.auto_attempts > 0) {
        println!(
            "  autopilot a/b = predictions within tolerance / scored. They show the cost model works; they don't pass lessons for you."
        );
    }
    if sim > 0 && real == 0 {
        println!(
            "  all evidence so far is SIMULATED — passing here means you understand the harness, not your hardware."
        );
    }
}

pub fn current_lesson(records: &[Record]) -> u8 {
    progress(records)
        .iter()
        .find(|p| !p.passed)
        .map(|p| p.n)
        .unwrap_or(6)
}

struct Facts {
    runs: Vec<Record>,
    agents: Vec<Record>,
    swept: BTreeSet<String>,
    labels: BTreeSet<String>,
    critic_modes: BTreeSet<String>,
    windows: BTreeSet<String>,
    agent_cache: bool,
    real: bool,
}

fn facts(records: &[Record]) -> Facts {
    let runs: Vec<Record> = records
        .iter()
        .filter(|r| r.kind == Kind::Run && !r.warmup)
        .cloned()
        .collect();
    let agents: Vec<Record> = records
        .iter()
        .filter(|r| r.kind == Kind::Agent)
        .cloned()
        .collect();
    Facts {
        swept: runs
            .iter()
            .filter(|r| r.experiment.starts_with("sweep:"))
            .filter_map(|r| r.var.clone())
            .collect(),
        labels: runs
            .iter()
            .chain(&agents)
            .filter_map(|r| r.label.clone())
            .collect(),
        critic_modes: agents
            .iter()
            .filter_map(|r| r.tags.get("critic").cloned())
            .collect(),
        windows: agents
            .iter()
            .filter_map(|r| r.tags.get("memory_window").cloned())
            .collect(),
        agent_cache: agents
            .iter()
            .any(|r| r.params.as_ref().is_some_and(|p| p.cache_prompt)),
        real: runs.iter().chain(&agents).any(|r| !r.simulated),
        runs,
        agents,
    }
}

fn group_cv(runs: &[Record], metric: &str) -> Option<(f64, usize)> {
    // Largest repeat group, most recent first.
    let mut groups: Vec<(&str, Vec<f64>)> = Vec::new();
    for r in runs.iter().rev().filter(|r| r.experiment == "repeat") {
        if let Some(v) = r.metrics.get(metric) {
            match groups.iter_mut().find(|g| g.0 == r.group) {
                Some(g) => g.1.push(*v),
                None => groups.push((&r.group, vec![*v])),
            }
        }
    }
    let g = groups.into_iter().find(|g| g.1.len() >= 3)?;
    let s = Summary::of(&g.1)?;
    Some((s.cv(), s.n))
}

/// A single-variable lever the notebook says is worth pulling, in the same
/// priority order `next` uses. Shared by `next` (prose) and the autopilot
/// (which turns each lever into an executable `ExperimentSpec`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lever {
    /// No measured runs yet.
    Baseline,
    /// No repeat group yet (cv = None), or the latest one is noisy (cv > 5%).
    NoiseFloor { cv: Option<f64> },
    /// A per-request variable that has never been swept.
    Sweep(&'static str),
    /// Quantization has never been compared. Needs a server restart: human only.
    QuantNeedsRestart,
    /// No agent episodes yet.
    AgentBaseline,
    /// A critic mode ("model" | "oracle") never tried.
    Critic(&'static str),
    /// Every agent run used full history.
    MemoryWindow,
    /// The agent never ran with prefix caching.
    AgentCache,
}

/// Sweep variables in the order `next` recommends them.
pub const SWEEP_ORDER: [&str; 4] = ["max_tokens", "prompt_tokens", "temperature", "cache_prompt"];

/// Every untried, high-value lever, highest priority first.
pub fn levers(records: &[Record]) -> Vec<Lever> {
    let f = facts(records);
    if f.runs.is_empty() {
        return vec![Lever::Baseline];
    }
    let mut out = Vec::new();
    match group_cv(&f.runs, "decode_tok_s") {
        None => out.push(Lever::NoiseFloor { cv: None }),
        Some((cv, _)) if cv > 0.05 => out.push(Lever::NoiseFloor { cv: Some(cv) }),
        _ => {}
    }
    out.extend(
        SWEEP_ORDER
            .iter()
            .filter(|v| !f.swept.contains(**v))
            .map(|v| Lever::Sweep(v)),
    );
    if !f.labels.iter().any(|l| l.contains('q')) && f.real {
        out.push(Lever::QuantNeedsRestart);
    }
    if f.agents.is_empty() {
        out.push(Lever::AgentBaseline);
    } else {
        for mode in ["model", "oracle"] {
            if !f.critic_modes.contains(mode) {
                out.push(Lever::Critic(mode));
            }
        }
        if !f.windows.iter().any(|w| w != "all") {
            out.push(Lever::MemoryWindow);
        }
        if !f.agent_cache {
            out.push(Lever::AgentCache);
        }
    }
    out
}

/// One concrete next experiment, with the number that motivates it.
pub fn next(records: &[Record]) -> (String, String) {
    let f = facts(records);
    let backend = if f.real {
        "--backend llama-server "
    } else {
        ""
    };
    if f.runs.is_empty() {
        return (
            "There are no measured runs yet. Establish the baseline: one completion, prefill vs decode.".into(),
            format!("machine run {backend}--lesson 1 --max-tokens 64"),
        );
    }
    match group_cv(&f.runs, "decode_tok_s") {
        None => {
            return (
                "You have single runs but no repeated group, so you don't know your noise floor yet.".into(),
                format!("machine run {backend}--lesson 2 --repeat 10 --warmup 1"),
            );
        }
        Some((cv, n)) if cv > 0.05 => {
            return (
                format!(
                    "Your latest repeat group (n={n}) has decode cv = {:.1}%. Any effect smaller than ~{:.0}% is invisible. Reduce noise before optimizing: close background apps, fix thread count (-t), add warm-up, then re-measure.",
                    cv * 100.0,
                    cv * 200.0
                ),
                format!("machine run {backend}--lesson 2 --repeat 15 --warmup 2 --label quiet"),
            );
        }
        _ => {}
    }
    for (var, why, cmd) in [
        (
            "max_tokens",
            "Output length is the cheapest knob to understand: its slope is your per-token decode cost.",
            "--var max-tokens --values 16,32,64,128",
        ),
        (
            "prompt_tokens",
            "You haven't measured how prefill scales with context. Agents grow their prompt every step, so this slope predicts agent cost.",
            "--var prompt-tokens --values 64,256,1024 --metric prefill_ms",
        ),
        (
            "temperature",
            "Greedy vs sampling hasn't been compared. Measure the speed cost, then (in lesson 5) the quality cost.",
            "--var temperature --values 0,0.8 --metric decode_tok_s --repeat 5",
        ),
        (
            "cache_prompt",
            "Prompt-prefix caching hasn't been measured. With a fixed prompt, a cache hit should collapse prefill_ms.",
            "--var cache-prompt --values false,true --metric prefill_ms --repeat 5",
        ),
    ] {
        if !f.swept.contains(var) {
            return (
                why.into(),
                if cmd.contains("--repeat") {
                    format!("machine sweep {backend}--lesson 3 {cmd}")
                } else {
                    format!("machine sweep {backend}--lesson 3 {cmd} --repeat 3")
                },
            );
        }
    }
    if !f.labels.iter().any(|l| l.contains('q')) && f.real {
        return (
            "All runs use one quantization. Load-time knobs need a server restart: compare two quants of the same model.".into(),
            "llama-server -m <model>-q8_0.gguf ...; machine run --backend llama-server --repeat 5 --warmup 1 --label q8 --tag quant=q8_0\nthen the same with q4_k_m and: machine compare --label q4 --label q8".into(),
        );
    }
    if f.agents.is_empty() {
        let per_call = Summary::of(
            &f.runs
                .iter()
                .filter_map(|r| r.metrics.get("wall_ms").copied())
                .collect::<Vec<_>>(),
        )
        .map(|s| s.mean)
        .unwrap_or(0.0);
        return (
            format!(
                "The single-call path is measured (mean wall {per_call:.0} ms/call). Wrap it in a loop: predict episode wall ≈ calls × that."
            ),
            format!("machine agent {backend}--lesson 4 --verbose"),
        );
    }
    if !f.critic_modes.contains("model") {
        return (
            "The agent runs without verification. Add a model critic and measure what it buys vs what it costs.".into(),
            format!("machine agent {backend}--lesson 5 --critic model --temperature 0.8 --label critic  (and --critic none --label nocritic)"),
        );
    }
    if !f.critic_modes.contains("oracle") {
        return (
            "You've tried a model critic but not the oracle — without the ceiling you can't tell how good your critic is.".into(),
            format!("machine agent {backend}--lesson 5 --critic oracle --temperature 0.8 --label oracle"),
        );
    }
    if !f.windows.iter().any(|w| w != "all") {
        return (
            "Every agent run used full history. Short-term memory is a single variable: shrink it and measure prompt_tokens vs success.".into(),
            format!("machine agent {backend}--lesson 6 --memory-window 1 --label win1  (compare with a full-history label)"),
        );
    }
    if !f.agent_cache {
        let ptok = Summary::of(
            &f.agents
                .iter()
                .filter_map(|r| {
                    Some(r.metrics.get("prompt_tokens")? / r.metrics.get("calls")?.max(1.0))
                })
                .collect::<Vec<_>>(),
        )
        .map(|s| s.mean)
        .unwrap_or(0.0);
        return (
            format!(
                "Your agent re-sends ~{ptok:.0} prompt tokens per call, and the instruction prefix never changes. Turn on prefix caching for the agent and predict how much prefill drops on calls 2+."
            ),
            format!(
                "machine agent {backend}--lesson 6 --cache-prompt --label agentcache  (compare with --label nocritic or your last full-history run)"
            ),
        );
    }
    (
        "Every built-in knob has been tried. Change the environment instead: add a second tool (e.g. a unit converter) and measure whether tool-choice errors cost more calls than they save.".into(),
        "edit src/agent.rs: add Action::Convert, one new task, then machine agent --verbose".into(),
    )
}

pub fn wrap_up(records: &[Record], learned: &[crate::learnings::Learning]) {
    println!("═══ SESSION CHECK ═══\n");
    println!("Answer these out loud or in writing before you close the terminal:");
    println!(
        "  1. Can you explain why the current loop is fast or slow without appealing to magic?"
    );
    println!(
        "     (name the phase — prefill, decode, overhead, extra calls — and the number that shows it)"
    );
    println!(
        "  2. Can you predict the effect of removing or changing one component before running it?"
    );
    println!("  3. Can you design the next single-variable experiment yourself?\n");

    let ps = progress(records);
    let weakest = ps.iter().filter(|p| p.attempts > 0).max_by(|a, b| {
        a.mean_error_pct
            .unwrap_or(0.0)
            .total_cmp(&b.mean_error_pct.unwrap_or(0.0))
    });
    match weakest {
        Some(w) if w.mean_error_pct.unwrap_or(0.0) > crate::predict::DEFAULT_TOLERANCE_PCT => {
            let l = lessons::get(w.n).unwrap();
            println!(
                "WEAKEST IDEA: lesson {} ({}) — mean prediction error {:.0}%.",
                w.n,
                l.title,
                w.mean_error_pct.unwrap_or(0.0)
            );
            println!("  Rebuild it smaller: {}", smaller_experiment(w.n));
        }
        Some(_) => println!(
            "No weak spot: every lesson you've predicted on is within tolerance on average."
        ),
        None => println!("No predictions recorded yet — run any experiment and predict first."),
    }

    if let Some(last) = learned.last() {
        let passed = learned.iter().filter(|l| l.passed).count();
        println!(
            "\nAUTOPILOT: {} trial(s), {passed} with every prediction within tolerance (`machine learnings`).",
            learned.len()
        );
        println!("  latest ({}): {}", last.key, last.learning);
        println!(
            "  Can you explain that result yourself, from the numbers, before reading the next one?"
        );
    }

    let (why, cmd) = next(records);
    println!("\nWHAT SHOULD I BUILD NEXT?");
    println!("  {why}");
    for line in cmd.lines() {
        println!("  $ {line}");
    }
}

fn smaller_experiment(n: u8) -> &'static str {
    match n {
        1 => {
            "fix the prompt, run --max-tokens 8 then 16 only; predict decode_ms from the 8-token run before running 16."
        }
        2 => {
            "run --repeat 5 twice in a row with no changes; predict whether the two means fall within each other's sd."
        }
        3 => {
            "sweep only two max_tokens values (32, 64) with --repeat 5; predict the second from the first assuming linear decode."
        }
        4 => {
            "run a single task (--tasks 0) with --verbose; predict prompt_tokens of call 2 from call 1 plus the history line length."
        }
        5 => {
            "run --critic model on --tasks 0 only; predict the critic's prefill share of its own call time."
        }
        _ => "pick one knob, two values, five repeats; write the prediction first.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook::PredictionResult;

    fn pred(lesson: u8, err: f64) -> Record {
        Record {
            id: "x".into(),
            ts: 0,
            kind: Kind::Prediction,
            group: "g".into(),
            lesson: Some(lesson),
            experiment: "single".into(),
            label: None,
            tags: Default::default(),
            backend: "mock".into(),
            model: None,
            params: None,
            var: None,
            value: None,
            metrics: Default::default(),
            simulated: true,
            predictions: vec![PredictionResult {
                metric: "wall_ms".into(),
                predicted: 1.0,
                actual: 1.0,
                error_pct: err,
                passed: err <= 20.0,
            }],
            text_preview: String::new(),
            warmup: false,
        }
    }

    #[test]
    fn gating() {
        let rs = vec![pred(1, 5.0), pred(1, 50.0)];
        assert!(!progress(&rs)[0].passed, "latest failed");
        let rs = vec![pred(1, 5.0), pred(1, 50.0), pred(1, 10.0)];
        assert!(progress(&rs)[0].passed);
        assert_eq!(current_lesson(&rs), 2);
    }

    #[test]
    fn levers_agree_with_next() {
        assert_eq!(levers(&[]), vec![Lever::Baseline]);
        let (_, cmd) = next(&[]);
        assert!(cmd.contains("machine run"));
    }

    #[test]
    fn next_starts_with_baseline() {
        let (_, cmd) = next(&[]);
        assert!(cmd.contains("--lesson 1"));
    }
}
