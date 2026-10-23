//! Fit everything the notebook can support and write it down as beliefs.
//! Pure: records (and learnings) in, `Analysis` out. No inference, no I/O.

use super::BASELINE_KEY;
use super::beliefs::{AgentConfig, Anomaly, Belief, BeliefState, FailureMode, Status};
use super::fit::{self, Est};
use super::surrogate::Surrogate;
use crate::agent::{self, CriticMode};
use crate::autopilot::AgentSettings;
use crate::learnings::{self, Learning};
use crate::notebook::{Kind, Record, now_secs};
use crate::stats::Summary;
use std::collections::{BTreeMap, BTreeSet};

pub struct Analysis {
    pub beliefs: BeliefState,
    pub surrogate: Surrogate,
}

/// A row is a prefix-cache hit when caching was on and the server evaluated
/// almost nothing (it reports evaluated, not total, prompt tokens).
pub fn is_cache_hit(r: &Record) -> bool {
    r.params.as_ref().is_some_and(|p| p.cache_prompt)
        && r.metrics.get("prompt_tokens").is_some_and(|t| *t <= 4.0)
}

fn runs(records: &[Record]) -> Vec<&Record> {
    records
        .iter()
        .filter(|r| r.kind == Kind::Run && !r.warmup)
        .collect()
}

fn col(rows: &[&Record], k: &str) -> Vec<f64> {
    rows.iter()
        .filter_map(|r| r.metrics.get(k).copied())
        .collect()
}

fn pairs(rows: &[&Record], x: &str, y: &str) -> Vec<(f64, f64)> {
    rows.iter()
        .filter_map(|r| Some((*r.metrics.get(x)?, *r.metrics.get(y)?)))
        .collect()
}

pub fn settings_of(r: &Record) -> Option<AgentSettings> {
    let critic = match r.tags.get("critic")?.as_str() {
        "none" => CriticMode::None,
        "model" => CriticMode::Model,
        "oracle" => CriticMode::Oracle,
        _ => return None,
    };
    let memory_window = match r.tags.get("memory_window")?.as_str() {
        "all" => None,
        w => Some(w.parse().ok()?),
    };
    let p = r.params.as_ref()?;
    Some(AgentSettings {
        critic,
        memory_window,
        cache_prompt: p.cache_prompt,
        temperature: p.temperature,
    })
}

pub fn config_key(s: &AgentSettings) -> String {
    format!(
        "critic={} window={} cache={} temp={}",
        s.critic_tag(),
        s.window_tag(),
        s.cache_prompt,
        s.temperature
    )
}

/// Agent rows from runs over the full task set, grouped by configuration.
fn agent_groups(records: &[Record]) -> BTreeMap<String, (AgentSettings, Vec<&Record>, usize)> {
    let mut by_group: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in records.iter().filter(|r| r.kind == Kind::Agent) {
        by_group.entry(&r.group).or_default().push(r);
    }
    let mut out: BTreeMap<String, (AgentSettings, Vec<&Record>, usize)> = BTreeMap::new();
    for rows in by_group.values() {
        if rows.len() < agent::TASKS.len() {
            continue; // subset runs (e.g. --tasks 0) would bias success
        }
        let Some(s) = settings_of(rows[0]) else {
            continue;
        };
        let e = out.entry(config_key(&s)).or_insert((s, Vec::new(), 0));
        e.1.extend(rows.iter().copied());
        e.2 += 1;
    }
    out
}

fn repeat_groups(records: &[Record]) -> Vec<Vec<&Record>> {
    let mut g: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in records
        .iter()
        .filter(|r| r.kind == Kind::Run && r.experiment == "repeat")
    {
        g.entry(&r.group).or_default().push(r);
    }
    g.into_values().collect()
}

pub const METHODS: &[&str] = &[
    "Lines (prefill vs evaluated prompt tokens, decode vs output tokens): Theil–Sen — median of pairwise slopes, intercept = median residual. Robust to warm-up outliers; no tuning constant.",
    "Intervals: 95% percentile bootstrap, 200 resamples, fixed seed 7. Same notebook ⇒ same beliefs.",
    "Success rates: Wilson score interval. Agent effects: difference of per-episode means vs the baseline config, two-sample bootstrap (unpaired; tasks are the same, so this is conservative).",
    "Agent configs use only runs over the full 10-task set; subset runs are excluded so success rates are comparable.",
    "Prefix-cache hit: cache_prompt=true and ≤4 evaluated prompt tokens. Modelled as paying only the fixed prefill cost.",
    "Status: parameters are KNOWN when the 95% half-width ≤ 15% of the estimate; effects are KNOWN when the interval excludes zero.",
    "Failure modes are rules over episode metrics, not clustering: solved; wrong answer (no format/tool errors: a planning error, since the calculator is exact); format loop (format errors until the step limit); format error then wrong answer; tool error.",
];

pub fn analyze(records: &[Record], learned: &[Learning], backend: &str) -> Analysis {
    let rows = runs(records);
    let simulated = records.iter().any(|r| r.simulated);
    let mut beliefs = Vec::new();

    // ── Cost model: prefill and decode lines ─────────────────────────────
    let prefill_pts = pairs(&rows, "prompt_tokens", "prefill_ms");
    let prefill = fit::fit_line(&prefill_pts);
    let decode_pts = pairs(&rows, "completion_tokens", "decode_ms");
    let decode = fit::fit_line(&decode_pts);
    let base_rows: Vec<&Record> = rows
        .iter()
        .copied()
        .filter(|r| r.var.is_none() && !is_cache_hit(r))
        .collect();
    let base_prompt_tokens = fit::median(&mut col(&base_rows, "prompt_tokens")).unwrap_or(30.0);
    let line_method = |l: &Option<fit::Line>| match l {
        Some(l) => format!(
            "Theil–Sen over {} rows at {} distinct lengths [{:.0}–{:.0}], R² {:.3}",
            l.n, l.distinct_x, l.x_min, l.x_max, l.r2
        ),
        None => "needs ≥3 rows at ≥2 distinct lengths".into(),
    };
    beliefs.push(
        Belief::new(
            "prefill.fixed_ms",
            "fixed prefill cost per call, paid even when ~0 tokens are evaluated",
            "ms",
            false,
            &line_method(&prefill),
        )
        .with(prefill.as_ref().map(|l| l.intercept), prefill_pts.len()),
    );
    beliefs.push(
        Belief::new(
            "prefill.ms_per_token",
            "prefill cost per evaluated prompt token",
            "ms",
            false,
            &line_method(&prefill),
        )
        .with(prefill.as_ref().map(|l| l.slope), prefill_pts.len()),
    );
    beliefs.push(
        Belief::new(
            "decode.ms_per_token",
            "decode cost per generated token",
            "ms",
            false,
            &line_method(&decode),
        )
        .with(decode.as_ref().map(|l| l.slope), decode_pts.len()),
    );
    beliefs.push(
        Belief::new(
            "decode.fixed_ms",
            "fixed decode cost per call",
            "ms",
            true, // a parameter whose natural null is 0: judge by "excludes zero"
            &line_method(&decode),
        )
        .with(decode.as_ref().map(|l| l.intercept), decode_pts.len()),
    );
    let ovh = col(&rows, "overhead_ms");
    let overhead = fit::mean_ci(&ovh);
    beliefs.push(
        Belief::new(
            "overhead.ms",
            "wall − prefill − decode per call (HTTP, JSON, scheduling)",
            "ms",
            true,
            "bootstrap mean over all measured calls",
        )
        .with(overhead, ovh.len()),
    );

    // ── Noise floor ───────────────────────────────────────────────────────
    let cvs: Vec<f64> = repeat_groups(records)
        .iter()
        .filter(|g| g.len() >= 5)
        .filter_map(|g| Summary::of(&col(g, "decode_tok_s")).map(|s| s.cv()))
        .collect();
    let noise = (!cvs.is_empty()).then(|| {
        let mut c = cvs.clone();
        let med = fit::median(&mut c).unwrap_or(0.0);
        Est {
            value: med,
            lo: c[0],
            hi: c[c.len() - 1],
        }
    });
    beliefs.push(
        Belief::new(
            "noise.decode_cv",
            "run-to-run spread of decode tok/s (median cv over repeat groups; interval = min–max)",
            "%",
            false,
            &format!("{} repeat groups with ≥5 runs", cvs.len()),
        )
        .with(noise, cvs.len()),
    );
    if cvs.len() < 3
        && let Some(b) = beliefs.last_mut()
        && b.status == Status::Known
    {
        b.status = Status::Uncertain; // a spread from <3 groups is an anecdote
    }

    // ── Effects of per-request knobs ──────────────────────────────────────
    let sweep_rows = |var: &str| -> Vec<&Record> {
        rows.iter()
            .copied()
            .filter(|r| r.experiment == format!("sweep:{var}"))
            .collect()
    };
    let at = |rs: &[&Record], v: &str, m: &str| -> Vec<f64> {
        rs.iter()
            .filter(|r| r.value.as_deref() == Some(v))
            .filter_map(|r| r.metrics.get(m).copied())
            .collect()
    };

    // Cache hit at the default prompt: groups whose cold rows are at the base length.
    let cache = sweep_rows("cache_prompt");
    let mut cache_groups: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in &cache {
        cache_groups.entry(&r.group).or_default().push(r);
    }
    // (full prompt length, hit prefill rows, miss prefill rows) per group
    let per_len: Vec<(f64, Vec<f64>, Vec<f64>)> = cache_groups
        .values()
        .filter_map(|g| {
            let mut cold = at(g, "false", "prompt_tokens");
            let full = fit::median(&mut cold)?;
            Some((
                full,
                at(g, "true", "prefill_ms"),
                at(g, "false", "prefill_ms"),
            ))
        })
        .collect();
    let near_base = |l: f64| (l - base_prompt_tokens).abs() <= base_prompt_tokens * 0.5;
    let (hits, misses): (Vec<f64>, Vec<f64>) = per_len.iter().filter(|(l, ..)| near_base(*l)).fold(
        (vec![], vec![]),
        |(mut h, mut m), (_, hh, mm)| {
            h.extend(hh);
            m.extend(mm);
            (h, m)
        },
    );
    beliefs.push(
        Belief::new(
            "effect.cache_hit.prefill_ms",
            "prefill change when the default prompt is already cached (cache_prompt false → true)",
            "Δms",
            true,
            "bootstrap difference of means, cache sweeps at the default prompt",
        )
        .with(fit::diff_ci(&misses, &hits), hits.len() + misses.len()),
    );
    // Interaction: does the hit cost depend on how long the cached prompt was?
    let lens: BTreeSet<i64> = per_len.iter().map(|(l, ..)| *l as i64).collect();
    let spread = lens
        .iter()
        .max()
        .zip(lens.iter().min())
        .map(|(a, b)| *a as f64 / (*b).max(1) as f64);
    let interaction = if spread.is_some_and(|s| s >= 2.0) {
        let pts: Vec<(f64, f64)> = per_len
            .iter()
            .flat_map(|(l, h, _)| h.iter().map(move |y| (*l, *y)))
            .collect();
        fit::fit_line(&pts).map(|l| l.slope)
    } else {
        None
    };
    beliefs.push(
        Belief::new(
            "interaction.cache_x_prompt",
            "extra cache-hit prefill per token of cached prompt (0 = a hit is equally cheap for long prompts)",
            "Δms",
            true,
            "slope of hit prefill vs cached prompt length across cache sweeps at ≥2 lengths (≥2× apart); ≈0 if under 10% of the per-token prefill cost",
        )
        .negligible_below(prefill.as_ref().map(|l| 0.1 * l.slope.value))
        .with(interaction, per_len.iter().map(|p| p.1.len()).sum()),
    );

    let temp = sweep_rows("temperature");
    let (t0, t8) = (
        at(&temp, "0", "decode_tok_s"),
        at(&temp, "0.8", "decode_tok_s"),
    );
    beliefs.push(
        Belief::new(
            "effect.temperature.decode_tok_s",
            "decode speed change from greedy to sampling (temperature 0 → 0.8)",
            "Δtok/s",
            true,
            "bootstrap difference of means, temperature sweeps; ≈0 if within 2× the decode noise floor",
        )
        .negligible_below(
            noise
                .zip(fit::mean_ci(&t0))
                .map(|(cv, m)| 2.0 * cv.value * m.value),
        )
        .with(fit::diff_ci(&t0, &t8), t0.len() + t8.len()),
    );

    // Quantization (load-time knob, only from human-tagged runs).
    let mut by_quant: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
    for r in &rows {
        if let Some(q) = r.tags.get("quant") {
            by_quant.entry(q.clone()).or_default().push(r);
        }
    }
    let quants: Vec<&String> = by_quant.keys().collect();
    for (m, unit, what) in [
        ("decode_tok_s", "Δtok/s", "decode speed"),
        ("peak_rss_mib", "ΔMiB", "server memory"),
    ] {
        let est = if quants.len() >= 2 {
            fit::diff_ci(&col(&by_quant[quants[0]], m), &col(&by_quant[quants[1]], m))
        } else {
            None
        };
        let label = if quants.len() >= 2 {
            format!("{} → {}", quants[0], quants[1])
        } else {
            "two tagged quantizations".into()
        };
        beliefs.push(
            Belief::new(
                &format!("effect.quant.{m}"),
                &format!(
                    "{what} change across quantizations ({label}); load-time knob, measured by you"
                ),
                unit,
                true,
                "bootstrap difference of means over runs tagged quant=…",
            )
            .with(est, by_quant.values().map(|v| v.len()).sum()),
        );
    }

    // ── Agent configurations and single-knob effects ──────────────────────
    let groups = agent_groups(records);
    let mut configs = Vec::new();
    for (key, (s, eps, runs)) in &groups {
        let succ = col(eps, "success");
        let k: f64 = succ.iter().sum();
        let calls = col(eps, "calls");
        let ptok: f64 = col(eps, "prompt_tokens").iter().sum();
        let ctot: f64 = calls.iter().sum();
        configs.push(AgentConfig {
            key: key.clone(),
            critic: s.critic_tag(),
            memory_window: s.window_tag(),
            cache_prompt: s.cache_prompt,
            temperature: s.temperature,
            episodes: eps.len(),
            runs: *runs,
            success: fit::wilson(k, succ.len() as f64),
            calls: fit::mean_ci(&calls).unwrap_or(Est {
                value: 0.0,
                lo: 0.0,
                hi: 0.0,
            }),
            wall_ms: fit::mean_ci(&col(eps, "wall_ms")).unwrap_or(Est {
                value: 0.0,
                lo: 0.0,
                hi: 0.0,
            }),
            prompt_per_call: if ctot > 0.0 { ptok / ctot } else { 0.0 },
            format_errors_per_task: col(eps, "format_errors").iter().sum::<f64>()
                / eps.len() as f64,
            critic_false_accepts: col(eps, "critic_false_accept").iter().sum(),
        });
    }
    let baseline = groups.get(BASELINE_KEY);
    beliefs.push(
        Belief::new(
            "agent.baseline.success",
            "success rate of the baseline agent (greedy, no critic, full history, no cache)",
            "rate",
            false,
            "Wilson 95% interval over 10-task runs",
        )
        .with(
            baseline.map(|(_, e, _)| {
                let s = col(e, "success");
                fit::wilson(s.iter().sum(), s.len() as f64)
            }),
            baseline.map(|b| b.1.len()).unwrap_or(0),
        ),
    );
    beliefs.push(
        Belief::new(
            "agent.baseline.wall_ms",
            "wall time per task of the baseline agent",
            "ms",
            false,
            "bootstrap mean over 10-task runs",
        )
        .with(
            baseline.and_then(|(_, e, _)| fit::mean_ci(&col(e, "wall_ms"))),
            baseline.map(|b| b.1.len()).unwrap_or(0),
        ),
    );
    // One belief per (knob, metric) for every config one setting away from baseline.
    let knobs: [(&str, AgentSettings); 5] = {
        let b = AgentSettings::BASELINE;
        [
            (
                "critic=model",
                AgentSettings {
                    critic: CriticMode::Model,
                    ..b
                },
            ),
            (
                "critic=oracle",
                AgentSettings {
                    critic: CriticMode::Oracle,
                    ..b
                },
            ),
            (
                "memory_window=1",
                AgentSettings {
                    memory_window: Some(1),
                    ..b
                },
            ),
            (
                "cache_prompt",
                AgentSettings {
                    cache_prompt: true,
                    ..b
                },
            ),
            (
                "temperature=0.8",
                AgentSettings {
                    temperature: 0.8,
                    ..b
                },
            ),
        ]
    };
    for (name, s) in knobs {
        let variant = groups.get(&config_key(&s));
        for (m, unit, what) in [
            ("success", "pts", "success"),
            ("calls", "Δcalls", "calls per task"),
            ("wall_ms", "Δms", "wall per task"),
        ] {
            let est = match (baseline, variant) {
                (Some((_, a, _)), Some((_, b, _))) => fit::diff_ci(&col(a, m), &col(b, m)),
                _ => None,
            };
            let n =
                baseline.map(|b| b.1.len()).unwrap_or(0) + variant.map(|v| v.1.len()).unwrap_or(0);
            beliefs.push(
                Belief::new(
                    &format!("effect.agent.{name}.{m}"),
                    &format!("{what} change from baseline when only {name} changes"),
                    unit,
                    true,
                    "bootstrap difference of per-episode means vs the baseline config",
                )
                .with(
                    est,
                    if est.is_some() {
                        n
                    } else {
                        variant.map(|v| v.1.len()).unwrap_or(0)
                    },
                ),
            );
        }
    }

    // ── Anomalies: assumptions the data contradicts ───────────────────────
    let mut anomalies = Vec::new();
    let hit_rows: Vec<&Record> = rows.iter().copied().filter(|r| is_cache_hit(r)).collect();
    let mut rates: Vec<f64> = rows
        .iter()
        .filter(|r| !is_cache_hit(r))
        .filter_map(|r| {
            let t = *r.metrics.get("prompt_tokens")?;
            (t >= 20.0).then(|| r.metrics.get("prefill_ms").map(|p| p / t))?
        })
        .collect();
    if let (Some(rate), Some(hit)) = (
        fit::median(&mut rates),
        Summary::of(&col(&hit_rows, "prefill_ms")),
    ) {
        let tokens = Summary::of(&col(&hit_rows, "prompt_tokens"))
            .map(|s| s.mean)
            .unwrap_or(1.0);
        let per_token_pred = rate * tokens;
        if hit.mean > 2.0 * per_token_pred {
            anomalies.push(Anomaly {
                key: "prefill.per_token_model_fails_on_cache_hits".into(),
                title: "A per-token-only prefill model fails on cache hits".into(),
                evidence: format!(
                    "{} cache-hit calls evaluated {tokens:.0} token(s) yet prefill took {:.1} ms on average; {rate:.2} ms/token predicts {per_token_pred:.1} ms ({:.0}× too low).",
                    hit.n,
                    hit.mean,
                    hit.mean / per_token_pred.max(1e-9)
                ),
                consequence: match &prefill {
                    Some(l) => format!(
                        "Model prefill as fixed + per-token: {:.1} ms + {:.2} ms/token. The fixed part bounds what caching can save per call.",
                        l.intercept.value, l.slope.value
                    ),
                    None => "Fit prefill with an intercept (fixed per-call cost).".into(),
                },
            });
        }
    }
    if let Some(d) = &decode
        && d.intercept.hi < 0.0
        && d.slope.value > 0.0
    {
        anomalies.push(Anomaly {
            key: "decode.negative_fixed_cost".into(),
            title: "Decode has a negative fixed cost".into(),
            evidence: format!(
                "decode_ms ≈ {:.1} ms + {:.2} ms × output tokens; the intercept is about {:.1} tokens' worth below zero.",
                d.intercept.value,
                d.slope.value,
                -d.intercept.value / d.slope.value
            ),
            consequence: "Physically a call can't save time, so the server's decode timer likely excludes some generated tokens (e.g. the first one, sampled at the end of prefill). Untested hypothesis; it doesn't affect the per-token slope, but don't read decode_ms/completion_tokens as exact for short outputs.".into(),
        });
    }
    if let Some(l) = &prefill
        && l.r2 < 0.9
        && l.distinct_x >= 3
    {
        anomalies.push(Anomaly {
            key: "prefill.not_linear".into(),
            title: "Prefill is not linear in prompt tokens over the measured range".into(),
            evidence: format!(
                "R² {:.3} over lengths {:.0}–{:.0}; typical miss {:.1} ms.",
                l.r2, l.x_min, l.x_max, l.mad_resid
            ),
            consequence:
                "Don't extrapolate prefill far beyond measured lengths; add points between them."
                    .into(),
        });
    }
    let early: Vec<&Record> = rows
        .iter()
        .copied()
        .filter(|r| {
            r.params.as_ref().is_some_and(|p| {
                !p.ignore_eos
                    && p.stop.is_empty()
                    && r.metrics
                        .get("completion_tokens")
                        .is_some_and(|c| *c < p.max_tokens as f64)
            })
        })
        .collect();
    if !early.is_empty() {
        anomalies.push(Anomaly {
            key: "decode.early_eos".into(),
            title: "Some runs stopped at end-of-sequence before max_tokens".into(),
            evidence: format!("{} run(s) generated fewer tokens than max_tokens without --ignore-eos.", early.len()),
            consequence: "Their wall_ms measures answer length, not speed. The decode line uses actual completion_tokens, so it is unaffected; per-run tok/s comparisons are not.".into(),
        });
    }
    let warm = repeat_groups(records)
        .iter()
        .filter(|g| g.len() >= 3 && !g.iter().any(|r| r.warmup))
        .filter(|g| {
            let w = col(g, "wall_ms");
            let mut rest = w[1..].to_vec();
            fit::median(&mut rest).is_some_and(|m| w[0] > 1.3 * m)
        })
        .count();
    if warm > 0 {
        anomalies.push(Anomaly {
            key: "noise.warmup_outliers".into(),
            title: "First run of a repeat group is a warm-up outlier".into(),
            evidence: format!(
                "{warm} repeat group(s) without --warmup had run #1 > 1.3× the median of the rest."
            ),
            consequence: "Always use --warmup ≥1; the robust lines already down-weight these rows."
                .into(),
        });
    }
    for c in configs
        .iter()
        .filter(|c| c.critic == "model" && c.critic_false_accepts > 0.0)
    {
        anomalies.push(Anomaly {
            key: "agent.critic_approves_wrong_answers".into(),
            title: "The model critic approves wrong answers".into(),
            evidence: format!(
                "{} false accept(s) over {} episodes ({}); success {:.0}% vs baseline {}.",
                c.critic_false_accepts,
                c.episodes,
                c.key,
                c.success.value * 100.0,
                baseline
                    .map(|(_, e, _)| format!("{:.0}%", col(e, "success").iter().sum::<f64>() / e.len() as f64 * 100.0))
                    .unwrap_or("unmeasured".into())
            ),
            consequence: "A same-model critic is not an independent check here; its calls are pure cost unless success rises.".into(),
        });
    }
    // Same configuration, different runs, different answers: something the
    // notebook doesn't record changed between them.
    for (key, (_, eps, runs)) in &groups {
        if *runs < 2 {
            continue;
        }
        let mut per: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
        for r in eps {
            let e = per.entry(&r.group).or_default();
            e.0 += r.metrics.get("wall_ms").copied().unwrap_or(0.0);
            e.1 += r.metrics.get("prompt_tokens").copied().unwrap_or(0.0)
                / r.metrics.get("calls").copied().unwrap_or(1.0).max(1.0);
            e.2 += 1;
        }
        let means: Vec<(f64, f64)> = per
            .values()
            .map(|(w, p, n)| (w / *n as f64, p / *n as f64))
            .collect();
        let (lo, hi) = means.iter().fold((f64::INFINITY, 0.0_f64), |(a, b), m| {
            (a.min(m.0), b.max(m.0))
        });
        if lo > 0.0 && hi / lo > 1.3 {
            let runs_txt: Vec<String> = means
                .iter()
                .map(|(w, p)| format!("{w:.0} ms/task at {p:.0} prompt tokens/call"))
                .collect();
            anomalies.push(Anomaly {
                key: format!("agent.irreproducible.{key}"),
                title: format!("Agent wall time did not reproduce across runs of {key}"),
                evidence: format!("{} runs: {}.", means.len(), runs_txt.join(" vs ")),
                consequence: "Something outside the recorded settings changed between runs (server state, background load, thermals). Pooled effects for this config carry that variance; replicate before trusting them.".into(),
            });
        }
    }
    let misses: Vec<&Learning> = learned
        .iter()
        .filter(|l| l.error_pct.values().any(|e| *e > 50.0))
        .collect();
    for l in misses.iter().take(5) {
        let (m, e) = l
            .error_pct
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(m, e)| (m.clone(), *e))
            .unwrap_or_default();
        anomalies.push(Anomaly {
            key: format!("prediction_miss.{}", l.key),
            title: format!("Autopilot prediction for {} missed by {e:.0}% on {m}", l.key),
            evidence: l.learning.clone(),
            consequence: "The model that made that prediction was wrong for this setup; the current beliefs refit from the new rows.".into(),
        });
    }

    // ── Failure modes (rules) and systematic task failures ────────────────
    let eps: Vec<&Record> = groups
        .values()
        .flat_map(|(_, e, _)| e.iter().copied())
        .collect();
    let max_steps = 6.0;
    let classify = |r: &Record| -> &'static str {
        let g = |k: &str| r.metrics.get(k).copied().unwrap_or(0.0);
        if g("success") > 0.0 {
            "solved"
        } else if g("tool_errors") > 0.0 {
            "tool error"
        } else if g("format_errors") > 0.0 && g("steps") >= max_steps {
            "format loop"
        } else if g("format_errors") > 0.0 {
            "format error then wrong answer"
        } else {
            "wrong answer (planning)"
        }
    };
    let rules: BTreeMap<&str, &str> = [
        (
            "tool error",
            "failed and the calculator rejected ≥1 expression",
        ),
        (
            "format loop",
            "failed, ≥1 format error, hit the 6-step limit",
        ),
        (
            "format error then wrong answer",
            "failed, ≥1 format error, answered before the limit",
        ),
        (
            "wrong answer (planning)",
            "failed with no format or tool errors — the plan (expression) was wrong",
        ),
    ]
    .into_iter()
    .collect();
    let failed: Vec<&Record> = eps
        .iter()
        .copied()
        .filter(|r| classify(r) != "solved")
        .collect();
    let mut modes: BTreeMap<&str, (usize, BTreeSet<String>)> = BTreeMap::new();
    for r in &failed {
        let e = modes.entry(classify(r)).or_default();
        e.0 += 1;
        if let Some(s) = settings_of(r) {
            e.1.insert(config_key(&s));
        }
    }
    let mut failure_modes: Vec<FailureMode> = modes
        .into_iter()
        .map(|(m, (n, cfgs))| FailureMode {
            mode: m.into(),
            rule: rules.get(m).copied().unwrap_or("").into(),
            episodes: n,
            share: n as f64 / failed.len().max(1) as f64,
            configs: cfgs.into_iter().collect(),
        })
        .collect();
    failure_modes.sort_by_key(|f| std::cmp::Reverse(f.episodes));
    let no_tool_field = eps
        .iter()
        .filter(|r| !r.metrics.contains_key("tool_errors"))
        .count();
    let mut methods: Vec<String> = METHODS.iter().map(|s| s.to_string()).collect();
    if no_tool_field > 0 {
        methods.push(format!(
            "{no_tool_field} agent episodes predate the tool_errors metric; their tool errors count as 0, so 'tool error' may be under-counted."
        ));
    }
    let mut task_results: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &eps {
        if let Some(t) = &r.value {
            let e = task_results.entry(t.clone()).or_default();
            e.0 += 1;
            if r.metrics.get("success").copied().unwrap_or(0.0) > 0.0 {
                e.1 += 1;
            }
        }
    }
    let n_cfg = groups.len();
    let systematic_failures: Vec<String> = task_results
        .iter()
        .filter(|(_, (n, ok))| n_cfg >= 3 && *n >= n_cfg && (*ok as f64) <= 0.1 * *n as f64)
        .map(|(t, (n, ok))| {
            let idx: usize = t.trim_start_matches("task").parse().unwrap_or(usize::MAX);
            let q = agent::TASKS.get(idx).map(|t| t.question).unwrap_or("?");
            format!("{t} solved {ok}/{n} episodes across all {n_cfg} configs: “{q}”")
        })
        .collect();

    let surrogate = Surrogate {
        prefill,
        decode,
        overhead,
        base_prompt_tokens,
        agent: configs.clone(),
    };
    Analysis {
        beliefs: BeliefState {
            version: 1,
            backend: backend.into(),
            simulated,
            fitted_at: learnings::iso8601(now_secs()),
            run_rows: rows.len(),
            agent_episodes: records.iter().filter(|r| r.kind == Kind::Agent).count(),
            beliefs,
            agent_configs: configs,
            anomalies,
            failure_modes,
            systematic_failures,
            methods,
        },
        surrogate,
    }
}

/// Human summary: what we know, what is uncertain, what broke.
pub fn print(a: &Analysis) {
    let b = &a.beliefs;
    println!("═══ ANALYSIS ═══  {}", b.backend);
    println!(
        "{} run rows, {} agent episodes, {} agent configurations{}",
        b.run_rows,
        b.agent_episodes,
        b.agent_configs.len(),
        if b.simulated {
            "  [SIMULATED — not evidence about hardware]"
        } else {
            ""
        }
    );

    println!("\n── COST MODEL (95% intervals)");
    let g = |k: &str| b.get(k).map(|x| x.fmt_est()).unwrap_or("—".into());
    println!(
        "  prefill  ≈ {} + {} × evaluated prompt tokens",
        g("prefill.fixed_ms"),
        g("prefill.ms_per_token")
    );
    println!(
        "  decode   ≈ {} + {} × output tokens",
        g("decode.fixed_ms"),
        g("decode.ms_per_token")
    );
    println!("  overhead ≈ {} per call", g("overhead.ms"));
    if let Some(t) = a.surrogate.decode_tok_s() {
        println!(
            "  ⇒ steady-state decode ≈ {t:.1} tok/s; noise floor (decode cv) {}",
            g("noise.decode_cv")
        );
    }
    if let Some(share) = a.surrogate.agent_prefill_share() {
        println!(
            "  ⇒ prefill is {:.0}% of a baseline agent call: prompt-side levers dominate agent cost",
            share * 100.0
        );
    }

    let show = |title: &str, st: Status| {
        let xs: Vec<_> = b.beliefs.iter().filter(|x| x.status == st).collect();
        println!("\n── {title} ({})", xs.len());
        for x in xs {
            println!("  {:<38} {:<34} {}", x.key, x.fmt_est(), x.what);
        }
    };
    show("WHAT WE NOW KNOW", Status::Known);
    show(
        "KNOWN TO BE ≈ ZERO (interval inside its practical-zero band)",
        Status::Negligible,
    );
    show(
        "STILL UNCERTAIN (measured, interval wide or includes 0)",
        Status::Uncertain,
    );
    show("UNTESTED", Status::Untested);

    if !b.anomalies.is_empty() {
        println!(
            "\n── BROKEN ASSUMPTIONS / ANOMALIES ({})",
            b.anomalies.len()
        );
        for x in &b.anomalies {
            println!("  ✗ {}", x.title);
            println!("    evidence: {}", x.evidence);
            println!("    so: {}", x.consequence);
        }
    }
    if !b.failure_modes.is_empty() {
        println!("\n── AGENT FAILURE MODES (rules, not clustering)");
        for f in &b.failure_modes {
            println!(
                "  {:<32} {:>3} episodes ({:>3.0}% of failures)  — {}",
                f.mode,
                f.episodes,
                f.share * 100.0,
                f.rule
            );
        }
        for s in &b.systematic_failures {
            println!("  systematic: {s}");
        }
    }
}
