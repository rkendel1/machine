//! Lessons 1–3: single completions, repeated runs, single-variable sweeps,
//! and side-by-side comparison of labelled runs.

use crate::backend::GenParams;
use crate::lab::{self, Lab};
use crate::notebook::{Kind, Record, new_id};
use crate::{metrics, stats};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub const DEFAULT_PROMPT: &str = "Explain in two sentences why generating text one token at a time is usually limited by memory bandwidth.";

/// Build a prompt of roughly `target` tokens by prefixing filler before `base`.
/// The server's own `prompt_tokens` is the truth; this is only an approximation.
pub fn padded_prompt(base: &str, target: usize) -> String {
    const FILLER: &str = "Note: this sentence is filler text used only to lengthen the context. ";
    let per = crate::backend::mock::approx_tokens(FILLER) as usize; // ≈ 18
    let base_tok = crate::backend::mock::approx_tokens(base) as usize;
    let n = target.saturating_sub(base_tok) / per.max(1);
    format!("{}{}", FILLER.repeat(n), base)
}

pub struct RunOpts {
    pub prompt: String,
    pub params: GenParams,
    pub repeat: usize,
    pub warmup: usize,
    pub pad_to: Option<usize>,
    pub show_text: bool,
}

const RUN_TABLE: &[&str] = &[
    "decode_tok_s",
    "prefill_tok_s",
    "prefill_ms",
    "decode_ms",
    "wall_ms",
    "overhead_ms",
    "prompt_tokens",
    "completion_tokens",
    "peak_rss_mib",
    "rss_delta_mib",
];

pub fn run(lab: &mut Lab, o: RunOpts) -> Result<lab::Outcome> {
    if o.repeat == 0 {
        bail!("--repeat must be ≥ 1");
    }
    let prompt = match o.pad_to {
        Some(n) => padded_prompt(&o.prompt, n),
        None => o.prompt.clone(),
    };
    let experiment = if o.repeat > 1 { "repeat" } else { "single" };
    lab.banner(&format!("RUN × {} (+{} warm-up)", o.repeat, o.warmup));
    println!(
        "params  : max_tokens={} temperature={} seed={:?} cache_prompt={} ignore_eos={}",
        o.params.max_tokens,
        o.params.temperature,
        o.params.seed,
        o.params.cache_prompt,
        o.params.ignore_eos
    );

    let mut ask = vec!["decode_tok_s", "wall_ms"];
    if lab.watch_pid.is_some() {
        ask.push("peak_rss_mib");
    }
    let preds = lab.collect_predictions(&ask, &[String::new()]);

    let group = new_id("grp");
    let mut rows = Vec::new();
    let mut texts = Vec::new();
    println!();
    for i in 0..(o.warmup + o.repeat) {
        let warm = i < o.warmup;
        let g = lab.infer(&prompt, &o.params)?;
        println!("{}", lab::one_line(i + 1, &g, warm));
        let m = metrics::from_generation(&g);
        let mut rec = lab.record(Kind::Run, &group, experiment);
        rec.params = Some(o.params.clone());
        rec.metrics = m.clone();
        rec.text_preview = g.text.chars().take(200).collect();
        rec.warmup = warm;
        if let Some(n) = o.pad_to {
            rec.var = Some("prompt_tokens".into());
            rec.value = Some(n.to_string());
        }
        lab.notebook.append(&rec)?;
        if !warm {
            rows.push(m);
            texts.push(g.text);
        }
    }

    if o.show_text || o.repeat == 1 {
        println!("\n── OUTPUT (first run) ─────────────────────────────────────────");
        println!("{}", texts[0].trim());
    }
    if texts.len() > 1 {
        let distinct = {
            let mut t = texts.clone();
            t.sort();
            t.dedup();
            t.len()
        };
        println!(
            "\ndistinct outputs: {distinct}/{} {}",
            texts.len(),
            if o.params.temperature == 0.0 && distinct > 1 {
                "← greedy but not identical: nondeterminism in the backend (batching, threads, GPU kernels)"
            } else {
                ""
            }
        );
    }

    println!(
        "\n── SUMMARY ({} measured runs) ─────────────────────────────────",
        rows.len()
    );
    lab::print_summary(&rows, RUN_TABLE);
    diagnose(&rows, o.warmup, Some(o.params.max_tokens));

    let mean = lab::means(&rows);
    let points = vec![String::new()];
    let results =
        lab.score_and_record(&group, experiment, &preds, &points, &[mean.clone()], None)?;
    Ok(lab::Outcome {
        group,
        points,
        means: vec![mean],
        results,
    })
}

/// Point at the specific numbers that usually explain surprises.
fn diagnose(rows: &[BTreeMap<String, f64>], warmup: usize, max_tokens: Option<u32>) {
    let get = |k: &str| -> Vec<f64> { rows.iter().filter_map(|r| r.get(k).copied()).collect() };
    let mut notes = Vec::new();
    if let Some(s) = stats::Summary::of(&get("decode_tok_s"))
        && s.n >= 3
        && s.cv() > 0.05
    {
        notes.push(format!(
                "decode_tok_s cv = {:.1}% (>5%): effects smaller than ~{:.0}% will hide in noise. Close other apps, pin threads, add --warmup, raise --repeat.",
                s.cv() * 100.0,
                2.0 * s.cv() * 100.0
            ));
    }
    let walls = get("wall_ms");
    if warmup == 0 && walls.len() >= 3 {
        let rest = stats::Summary::of(&walls[1..]).unwrap();
        if walls[0] > rest.p50 * 1.3 {
            notes.push(format!(
                "run #1 wall {:.0} ms vs median {:.0} ms of the rest: a warm-up effect (page-in of mmapped weights, model load, caches). Use --warmup 1 so it doesn't bias the mean.",
                walls[0], rest.p50
            ));
        }
    }
    let ctok = get("completion_tokens");
    if let Some(max) = max_tokens
        && !ctok.is_empty()
        && ctok.iter().any(|c| (*c as u32) < max)
    {
        notes.push(format!(
            "completion_tokens < max_tokens ({max}): the model hit end-of-sequence early. wall_ms then measures how long the ANSWER is, not how fast decoding is. For timing experiments use --ignore-eos (llama-server)."
        ));
    }
    let ptok = get("prompt_tokens");
    if ptok.len() >= 2 && ptok[1..].iter().any(|p| *p < ptok[0] * 0.5) {
        notes.push("prompt_tokens dropped after run #1: the server reused a cached prompt prefix. Prefill numbers are no longer comparable — use --no-cache-prompt (llama-server) or vary the prompt.".into());
    }
    if let Some(o) = stats::Summary::of(&get("overhead_ms"))
        && let Some(w) = stats::Summary::of(&walls)
        && o.mean > 0.2 * w.mean
    {
        notes.push(format!(
                    "overhead is {:.0}% of wall time: time is going somewhere other than prefill/decode (model load, HTTP, queueing). Check load_ms.",
                    100.0 * o.mean / w.mean
                ));
    }
    if !notes.is_empty() {
        println!("\n── WHAT THE NUMBERS ARE TELLING YOU");
        for n in notes {
            println!("  • {n}");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SweepVar {
    MaxTokens,
    Temperature,
    PromptTokens,
    Seed,
    CachePrompt,
}

impl SweepVar {
    pub fn name(self) -> &'static str {
        match self {
            SweepVar::MaxTokens => "max_tokens",
            SweepVar::Temperature => "temperature",
            SweepVar::PromptTokens => "prompt_tokens",
            SweepVar::Seed => "seed",
            SweepVar::CachePrompt => "cache_prompt",
        }
    }
    /// Spelling on the command line (`--var max-tokens`).
    pub fn cli_name(self) -> &'static str {
        match self {
            SweepVar::MaxTokens => "max-tokens",
            SweepVar::Temperature => "temperature",
            SweepVar::PromptTokens => "prompt-tokens",
            SweepVar::Seed => "seed",
            SweepVar::CachePrompt => "cache-prompt",
        }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        [
            SweepVar::MaxTokens,
            SweepVar::Temperature,
            SweepVar::PromptTokens,
            SweepVar::Seed,
            SweepVar::CachePrompt,
        ]
        .into_iter()
        .find(|v| v.name() == name)
    }
    pub fn default_metric(self) -> &'static str {
        match self {
            SweepVar::MaxTokens => "wall_ms",
            SweepVar::Temperature => "decode_tok_s",
            SweepVar::PromptTokens => "prefill_ms",
            SweepVar::Seed => "decode_tok_s",
            SweepVar::CachePrompt => "prefill_ms",
        }
    }
}

pub struct SweepOpts {
    pub var: SweepVar,
    pub values: Vec<String>,
    pub base: RunOpts,
    pub metric: Option<String>,
}

pub fn sweep(lab: &mut Lab, o: SweepOpts) -> Result<lab::Outcome> {
    if o.values.len() < 2 {
        bail!("a sweep needs at least two values");
    }
    let metric = o
        .metric
        .clone()
        .unwrap_or_else(|| o.var.default_metric().to_string());
    if !metrics::is_known(&metric) {
        bail!("unknown metric {metric}");
    }
    // Validate and build per-point (prompt, params) up front.
    let mut points = Vec::new();
    for v in &o.values {
        let mut p = o.base.params.clone();
        let mut prompt = match o.base.pad_to {
            Some(n) => padded_prompt(&o.base.prompt, n),
            None => o.base.prompt.clone(),
        };
        match o.var {
            SweepVar::MaxTokens => p.max_tokens = v.parse()?,
            SweepVar::Temperature => p.temperature = v.parse()?,
            SweepVar::Seed => p.seed = Some(v.parse()?),
            SweepVar::CachePrompt => p.cache_prompt = v.parse()?,
            SweepVar::PromptTokens => prompt = padded_prompt(&o.base.prompt, v.parse()?),
        }
        points.push((v.clone(), prompt, p));
    }

    lab.banner(&format!(
        "SWEEP {} ∈ {{{}}} × {} runs (+{} warm-up), interleaved",
        o.var.name(),
        o.values.join(", "),
        o.base.repeat,
        o.base.warmup
    ));
    println!(
        "only {} changes; everything else is held at the baseline.",
        o.var.name()
    );
    let labels: Vec<String> = o
        .values
        .iter()
        .map(|v| format!("{}={v}", o.var.name()))
        .collect();
    let preds = lab.collect_predictions(&[metric.as_str()], &labels);

    let group = new_id("grp");
    let mut rows: Vec<Vec<BTreeMap<String, f64>>> = vec![Vec::new(); points.len()];

    // Warm-up once on the first point, then round-robin so slow drift
    // (thermal throttling, background load) spreads evenly across values.
    for w in 0..o.base.warmup {
        let (_, prompt, p) = &points[0];
        let g = lab.infer(prompt, p)?;
        println!("{}", lab::one_line(w + 1, &g, true));
    }
    for r in 0..o.base.repeat {
        for (i, (v, prompt, p)) in points.iter().enumerate() {
            let g = lab.infer(prompt, p)?;
            if !lab.quiet {
                println!(
                    "{:>14} {}",
                    format!("{}={v}", o.var.name()),
                    lab::one_line(r + 1, &g, false)
                );
            }
            let m = metrics::from_generation(&g);
            let mut rec: Record = lab.record(Kind::Run, &group, &format!("sweep:{}", o.var.name()));
            rec.params = Some(p.clone());
            rec.var = Some(o.var.name().into());
            rec.value = Some(v.clone());
            rec.metrics = m.clone();
            rec.text_preview = g.text.chars().take(120).collect();
            lab.notebook.append(&rec)?;
            rows[i].push(m);
        }
    }

    println!(
        "\n── SWEEP RESULT: {metric} by {} ───────────────────────────",
        o.var.name()
    );
    let first = stats::Summary::of(&col(&rows[0], &metric));
    println!(
        "  {:<16} {:>10} {:>9} {:>6} {:>9} {:>9}  vs first value",
        o.var.name(),
        "mean",
        "sd",
        "cv%",
        "prompt_t",
        "compl_t"
    );
    for (i, (v, _, _)) in points.iter().enumerate() {
        let s = stats::Summary::of(&col(&rows[i], &metric));
        let pt = stats::Summary::of(&col(&rows[i], "prompt_tokens"))
            .map(|s| s.mean)
            .unwrap_or(0.0);
        let ct = stats::Summary::of(&col(&rows[i], "completion_tokens"))
            .map(|s| s.mean)
            .unwrap_or(0.0);
        if let Some(s) = s {
            let cmp = if i == 0 {
                "baseline".to_string()
            } else {
                let t = first.as_ref().and_then(|f| stats::welch_t(f, &s));
                let pct = first
                    .map(|f| 100.0 * (s.mean - f.mean) / f.mean)
                    .unwrap_or(0.0);
                format!("{pct:+.1}%  {}", stats::verdict(t))
            };
            println!(
                "  {:<16} {:>10} {:>9} {:>6.1} {:>9.0} {:>9.0}  {cmp}",
                v,
                metrics::short(s.mean),
                metrics::short(s.sd),
                s.cv() * 100.0,
                pt,
                ct
            );
        }
    }
    slope_note(
        o.var,
        &points.iter().map(|p| p.0.clone()).collect::<Vec<_>>(),
        &rows,
        &metric,
    );

    let means: Vec<_> = rows.iter().map(|r| lab::means(r)).collect();
    let results = lab.score_and_record(
        &group,
        &format!("sweep:{}", o.var.name()),
        &preds,
        &labels,
        &means,
        Some(o.var.name()),
    )?;
    Ok(lab::Outcome {
        group,
        points: labels,
        means,
        results,
    })
}

fn col(rows: &[BTreeMap<String, f64>], k: &str) -> Vec<f64> {
    rows.iter().filter_map(|r| r.get(k).copied()).collect()
}

/// For numeric sweeps, fit metric ≈ a + b·x by least squares and print b.
/// A straight line is a model you can predict with next time.
fn slope_note(var: SweepVar, values: &[String], rows: &[Vec<BTreeMap<String, f64>>], metric: &str) {
    if matches!(var, SweepVar::CachePrompt | SweepVar::Seed) {
        return;
    }
    let pts: Vec<(f64, f64)> = values
        .iter()
        .zip(rows)
        .filter_map(|(v, r)| {
            // For prompt length, fit against what the server actually counted,
            // not the padding target (the tokenizer decides the real length).
            let x = if matches!(var, SweepVar::PromptTokens) {
                stats::Summary::of(&col(r, "prompt_tokens"))?.mean
            } else {
                v.parse::<f64>().ok()?
            };
            Some((x, stats::Summary::of(&col(r, metric))?.mean))
        })
        .collect();
    if pts.len() < 3 {
        return; // two points always fit a line perfectly; R² would mean nothing
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx == 0.0 {
        return;
    }
    let b = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx;
    let a = my - b * mx;
    let ss_tot: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
    let ss_res: f64 = pts.iter().map(|p| (p.1 - (a + b * p.0)).powi(2)).sum();
    let r2 = if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };
    println!(
        "\n  linear fit: {metric} ≈ {:.3} + {:.4} × {}   (R² = {:.3})",
        a,
        b,
        if matches!(var, SweepVar::PromptTokens) {
            "measured prompt_tokens"
        } else {
            var.name()
        },
        r2
    );
    println!(
        "  use this line to predict the next point BEFORE you run it; R² well below 1 means the relationship isn't linear here."
    );
}

/// The settings a row was run with, as `key=value` strings: tags (load-time
/// knobs, agent settings), per-request params, and the backend. The
/// `source` tag (human vs autopilot) is provenance, not a setting.
pub fn settings(r: &Record) -> std::collections::BTreeSet<String> {
    let mut out: std::collections::BTreeSet<String> = r
        .tags
        .iter()
        .filter(|(k, _)| k.as_str() != "source")
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if let Some(p) = &r.params {
        out.insert(format!("max_tokens={}", p.max_tokens));
        out.insert(format!("temp={}", p.temperature));
        out.insert(format!("cache={}", p.cache_prompt));
    }
    out.insert(format!("backend={}", r.backend));
    out
}

/// Settings present in one row but not the other (both sides listed).
pub fn differing_settings(a: &Record, b: &Record) -> Vec<String> {
    let (a, b) = (settings(a), settings(b));
    a.symmetric_difference(&b).cloned().collect()
}

/// One changed setting shows up twice (old and new value).
pub fn single_variable(diff: &[String]) -> bool {
    diff.len() <= 2
}

/// Compare groups of runs by label (or tag) side by side.
pub fn compare(records: &[Record], labels: &[String], metric_names: &[String]) -> Result<()> {
    if labels.len() < 2 {
        bail!("compare needs at least two --label values (e.g. --label q4 --label q8)");
    }
    let names: Vec<String> = if metric_names.is_empty() {
        ["decode_tok_s", "prefill_tok_s", "wall_ms", "peak_rss_mib"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        metric_names.to_vec()
    };
    let groups: Vec<Vec<&Record>> = labels
        .iter()
        .map(|l| {
            records
                .iter()
                .filter(|r| {
                    matches!(r.kind, Kind::Run | Kind::Agent)
                        && !r.warmup
                        && r.label.as_deref() == Some(l)
                })
                .collect()
        })
        .collect();
    for (l, g) in labels.iter().zip(&groups) {
        if g.is_empty() {
            bail!("no runs with label {l:?}. Run with --label {l} first.");
        }
    }
    println!("═══ COMPARE ═══");
    for (l, g) in labels.iter().zip(&groups) {
        let desc: Vec<String> = settings(g[0]).into_iter().collect();
        println!(
            "  {l:<12} n={:<3} {}{}",
            g.len(),
            desc.join(" "),
            if g[0].simulated { "  [SIMULATED]" } else { "" }
        );
    }
    for (l, g) in labels.iter().zip(&groups).skip(1) {
        let diff = differing_settings(groups[0][0], g[0]);
        if !single_variable(&diff) {
            println!(
                "  ⚠ {l} differs from {} in more than one setting: {diff:?} — causal attribution is unclear.",
                labels[0]
            );
        }
    }
    println!();
    print!("  {:<18}", "metric");
    for l in labels {
        print!(" {:>16}", l);
    }
    println!("  Δ last vs first");
    for name in &names {
        let sums: Vec<Option<stats::Summary>> = groups
            .iter()
            .map(|g| {
                stats::Summary::of(
                    &g.iter()
                        .filter_map(|r| r.metrics.get(name).copied())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        if sums.iter().all(Option::is_none) {
            continue;
        }
        print!("  {name:<18}");
        for s in &sums {
            match s {
                Some(s) => print!(
                    " {:>16}",
                    format!("{}±{}", metrics::short(s.mean), metrics::short(s.sd))
                ),
                None => print!(" {:>16}", "—"),
            }
        }
        if let (Some(Some(a)), Some(Some(b))) = (sums.first(), sums.last()) {
            let delta = if metrics::unit(name) == "0/1" {
                format!("{:+.0} pts", 100.0 * (b.mean - a.mean))
            } else if a.mean != 0.0 {
                format!("{:+.1}%", 100.0 * (b.mean - a.mean) / a.mean)
            } else {
                format!("{:+} abs", metrics::short(b.mean - a.mean))
            };
            let better = if b.mean == a.mean {
                "same"
            } else if metrics::higher_is_better(name) == (b.mean > a.mean) {
                "better"
            } else {
                "worse"
            };
            print!(
                "  {delta} ({better}), {}",
                stats::verdict(stats::welch_t(a, b))
            );
        }
        println!();
    }
    Ok(())
}
