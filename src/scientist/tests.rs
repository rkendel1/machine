//! Scientist tests: the real example notebooks (fixtures checked into
//! `examples/`) and the full loop on the instant mock.

use super::analyze::{self, Analysis};
use super::beliefs::Status;
use super::selector::{self, Scientist};
use super::surrogate::CallQuery;
use crate::autopilot::{self, Approval, AutopilotState, test_support};
use crate::learnings::{self, Learning};
use crate::notebook::Record;

const REAL: &str = "llama-server@http://127.0.0.1:8080";
const AFTER_AUTOPILOT: &str = include_str!("../../examples/autopilot/notebook.jsonl");
const AFTER_AUTOPILOT_LEARNINGS: &str = include_str!("../../examples/autopilot/learnings.jsonl");
const BEFORE_AUTOPILOT: &str = include_str!("../../examples/notebook-2vcpu-qwen0.5b.jsonl");

fn parse<T: serde::de::DeserializeOwned>(s: &str) -> Vec<T> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn real() -> (Vec<Record>, Vec<Learning>, Analysis) {
    let r: Vec<Record> = autopilot::relevant(&parse(AFTER_AUTOPILOT), REAL);
    let l: Vec<Learning> = parse(AFTER_AUTOPILOT_LEARNINGS);
    let a = analyze::analyze(&r, &l, REAL);
    (r, l, a)
}

fn est(a: &Analysis, k: &str) -> super::fit::Est {
    a.beliefs
        .get(k)
        .unwrap_or_else(|| panic!("no belief {k}"))
        .est
        .unwrap_or_else(|| panic!("{k} untested"))
}

#[test]
fn surfaces_the_known_lessons_from_the_real_notebook() {
    let (_, _, a) = real();
    let anomaly = |k: &str| a.beliefs.anomalies.iter().any(|x| x.key == k);

    // Cache hits pay a fixed prefill cost a per-token model can't see.
    assert!(anomaly("prefill.per_token_model_fails_on_cache_hits"));
    let fixed = est(&a, "prefill.fixed_ms");
    assert!(
        fixed.value > 15.0 && fixed.value < 60.0 && fixed.lo > 0.0,
        "{fixed:?}"
    );
    assert!(est(&a, "effect.cache_hit.prefill_ms").hi < 0.0);

    // Agent prefix caching is a real, large saving.
    let cache = a.beliefs.get("effect.agent.cache_prompt.wall_ms").unwrap();
    assert_eq!(cache.status, Status::Known);
    assert!(cache.est.unwrap().value < -1500.0);

    // Memory window 1 regressed (direction), but n is small: still uncertain.
    let mw = a
        .beliefs
        .get("effect.agent.memory_window=1.success")
        .unwrap();
    assert!(mw.est.unwrap().value < 0.0);
    assert_eq!(mw.status, Status::Uncertain);
    assert!(est(&a, "effect.agent.memory_window=1.calls").value > 0.0);

    // The model critic doesn't help and approves wrong answers.
    let critic = est(&a, "effect.agent.critic=model.success");
    assert!(critic.lo <= 0.0 && critic.hi >= 0.0);
    assert!(anomaly("agent.critic_approves_wrong_answers"));

    // Sampling hurts the agent; quantization effects are known; interaction untested.
    assert!(est(&a, "effect.agent.temperature=0.8.success").hi < 0.0);
    assert_eq!(
        a.beliefs.get("effect.quant.peak_rss_mib").unwrap().status,
        Status::Known
    );
    assert_eq!(
        a.beliefs.get("interaction.cache_x_prompt").unwrap().status,
        Status::Untested
    );
    assert!(!a.beliefs.systematic_failures.is_empty());
    assert!(
        a.beliefs
            .failure_modes
            .iter()
            .any(|f| f.mode == "format loop")
    );
}

#[test]
fn beliefs_are_deterministic_and_serializable() {
    let (_, _, a) = real();
    let (_, _, b) = real();
    let strip = |a: &Analysis| {
        let mut v = serde_json::to_value(&a.beliefs).unwrap();
        v["fitted_at"] = serde_json::Value::Null;
        v
    };
    assert_eq!(strip(&a), strip(&b));
    let back: super::beliefs::BeliefState =
        serde_json::from_value(serde_json::to_value(&a.beliefs).unwrap()).unwrap();
    assert_eq!(back.beliefs.len(), a.beliefs.beliefs.len());
}

#[test]
fn surrogate_prices_a_cache_hit_at_the_fixed_cost() {
    let (_, _, a) = real();
    let q = CallQuery::parse("prompt_tokens=512 max_tokens=8 cache=true", 28.0).unwrap();
    let hit = a.surrogate.call(&q).unwrap();
    let fixed = est(&a, "prefill.fixed_ms");
    assert!((hit.prefill_ms.value - fixed.value).abs() < 10.0);
    let cold = a
        .surrogate
        .call(&CallQuery {
            cache_hit: false,
            ..q
        })
        .unwrap();
    assert!(cold.prefill_ms.value > 20.0 * hit.prefill_ms.value);
    assert!(cold.wall_ms.lo <= cold.wall_ms.value && cold.wall_ms.value <= cold.wall_ms.hi);
}

#[test]
fn selector_tests_the_untested_interaction_first() {
    let (r, _, a) = real();
    let ranked = selector::rank(&a, &r, &AutopilotState::new(1e9, 100, None));
    let top = &ranked[0];
    assert_eq!(top.proposal.key, "interaction:cache_x_prompt");
    top.proposal.spec.validate().unwrap();
    let cmd = top.proposal.command("--backend llama-server ");
    assert!(
        cmd.contains("--var cache-prompt")
            && cmd.contains("--pad-to 512")
            && cmd.contains("--predict prefill_ms="),
        "{cmd}"
    );
    // Everything it ranks is a validated single-variable spec with predictions.
    for c in &ranked {
        c.proposal.spec.validate().unwrap();
        assert!(!c.proposal.predictions.is_empty(), "{}", c.proposal.key);
    }
}

#[test]
fn orders_better_than_the_checklist_on_moderate_data() {
    // Before the autopilot ran, the checklist would do memory_window before
    // agent caching. Prefill is ~78% of an agent call, so the selector puts
    // agent caching (the lever on that share) first.
    let r: Vec<Record> = autopilot::relevant(&parse(BEFORE_AUTOPILOT), REAL);
    let a = analyze::analyze(&r, &[], REAL);
    assert!(!selector::sparse(&a));
    let st = AutopilotState::new(1e9, 100, None);
    let keys: Vec<String> = selector::rank(&a, &r, &st)
        .into_iter()
        .map(|c| c.proposal.key)
        .collect();
    let pos = |k: &str| keys.iter().position(|x| x == k).unwrap();
    assert!(
        pos("agent:cache_prompt") < pos("agent:memory_window=1"),
        "{keys:?}"
    );

    let mut checklist = Vec::new();
    let mut s = st.clone();
    while let Some(p) = autopilot::propose_next(&s, &r) {
        s.tried.insert(p.key.clone());
        checklist.push(p.key);
    }
    let cpos = |k: &str| checklist.iter().position(|x| x == k).unwrap();
    assert!(cpos("agent:memory_window=1") < cpos("agent:cache_prompt"));
}

#[test]
fn report_explains_beliefs() {
    let (r, l, a) = real();
    let ranked = selector::rank(&a, &r, &AutopilotState::new(1e9, 100, None));
    let md = super::report::markdown(&a, &ranked, &l, "--backend llama-server ");
    for section in [
        "## Summary",
        "## Cost model",
        "## What we now know",
        "## What is still uncertain",
        "## Broken assumptions",
        "## Recommended next experiments",
        "## Methods and limitations",
    ] {
        assert!(md.contains(section), "missing {section}");
    }
    assert!(md.contains("ms fixed +"));
    assert!(md.contains("interaction:cache_x_prompt"));
}

#[test]
fn scientist_loop_runs_on_the_mock() {
    let _g = test_support::LOOP.lock().unwrap_or_else(|e| e.into_inner());
    autopilot::interrupt::install();
    let (mut lab, dir) = test_support::mock_lab("scientist");
    let name = lab.backend.name();
    let mut planner = Scientist::new(&name, Some(dir.clone()));
    let opts = test_support::opts(9, None);
    let s =
        autopilot::execute_planned(&mut lab, &opts, &mut |_| Approval::Run, &mut planner).unwrap();
    assert!(!s.trials.is_empty());
    // Sparse at first → the checklist's baseline; then beliefs take over.
    assert_eq!(s.trials[0].0.key, "baseline");
    assert!(s.trials[0].0.rationale.starts_with("[fallback"));
    assert!(
        s.trials
            .iter()
            .any(|(p, _)| p.rationale.starts_with("[score")),
        "selector never took over"
    );
    for (p, o) in &s.trials {
        p.spec.validate().unwrap();
        assert!(!o.results.is_empty(), "{} was not scored", p.key);
    }
    // Explicit state on disk after every trial; learnings as usual.
    assert!(dir.join("beliefs.json").exists());
    let history = std::fs::read_to_string(dir.join("beliefs_history.jsonl")).unwrap();
    assert_eq!(history.lines().count(), s.trials.len());
    assert_eq!(
        learnings::load(&lab.notebook).unwrap().len(),
        s.trials.len()
    );
    let _ = std::fs::remove_dir_all(dir);
}
