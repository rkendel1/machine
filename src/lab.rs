//! Shared experiment context: one backend, one notebook, one memory watcher.
//! Every inference call goes through `Lab::infer` so every call is measured.

use crate::backend::{self, Backend, BackendConfig, BackendKind, GenParams, Generation};
use crate::memory::{self, Sampler, WatchTarget};
use crate::notebook::{Kind, Notebook, Record, new_id, now_secs};
use crate::predict::{self, Prediction};
use crate::{metrics, stats};
use anyhow::{Result, bail};
use std::collections::BTreeMap;
use std::time::Duration;

pub struct LabOptions {
    pub backend: BackendConfig,
    pub watch: String,
    pub label: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub lesson: Option<u8>,
    pub tolerance_pct: f64,
    pub predictions: Vec<String>,
    pub no_predict: bool,
    pub quiet: bool,
    pub template: Template,
}

/// How the prompt is wrapped before it reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Template {
    /// Send the prompt exactly as written.
    Raw,
    /// Wrap as a single ChatML user turn (Qwen, many instruct models).
    Chatml,
}

impl Template {
    pub fn apply(self, prompt: &str) -> String {
        match self {
            Template::Raw => prompt.to_string(),
            Template::Chatml => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }
}

pub struct Lab {
    pub backend: Box<dyn Backend>,
    pub kind: BackendKind,
    pub model_info: Option<String>,
    pub watch_pid: Option<u32>,
    pub watch_desc: String,
    pub notebook: Notebook,
    pub label: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub lesson: Option<u8>,
    pub tolerance_pct: f64,
    pub cli_predictions: Vec<Prediction>,
    pub no_predict: bool,
    pub quiet: bool,
    pub template: Template,
}

pub fn parse_watch(s: &str, kind: BackendKind, hint: Option<&'static str>) -> WatchTarget {
    match s {
        "none" => WatchTarget::None,
        "self" => WatchTarget::SelfProcess,
        "auto" => match (kind, hint) {
            (BackendKind::Mock, _) => WatchTarget::SelfProcess,
            (_, Some(h)) => WatchTarget::Name(h.to_string()),
            _ => WatchTarget::None,
        },
        s if s.starts_with("pid:") => s[4..]
            .parse()
            .map(WatchTarget::Pid)
            .unwrap_or(WatchTarget::None),
        s if s.starts_with("name:") => WatchTarget::Name(s[5..].to_string()),
        _ => WatchTarget::None,
    }
}

impl Lab {
    pub fn new(o: LabOptions) -> Result<Self> {
        let mut backend = backend::build(&o.backend)?;
        let model_info = backend.model_info();
        let target = parse_watch(&o.watch, o.backend.kind, backend.server_process_hint());
        let (watch_pid, watch_desc) = match memory::resolve(&target) {
            Some((pid, d)) => (Some(pid), d),
            None => {
                if !matches!(target, WatchTarget::None) {
                    eprintln!(
                        "note: could not find the process holding the model ({target:?}); memory will not be measured.\n      pass --watch pid:<PID> (find it with `pgrep -a llama-server` or `pgrep -a ollama`)."
                    );
                }
                (None, "not watched".into())
            }
        };
        let cli_predictions = o
            .predictions
            .iter()
            .map(|s| predict::parse(s))
            .collect::<Result<Vec<_>>>()?;
        if o.tolerance_pct <= 0.0 {
            bail!("--tolerance must be > 0");
        }
        Ok(Self {
            backend,
            kind: o.backend.kind,
            model_info,
            watch_pid,
            watch_desc,
            notebook: Notebook::open()?,
            label: o.label,
            tags: o.tags,
            lesson: o.lesson,
            tolerance_pct: o.tolerance_pct,
            cli_predictions,
            no_predict: o.no_predict,
            quiet: o.quiet,
            template: o.template,
        })
    }

    pub fn simulated(&self) -> bool {
        self.kind == BackendKind::Mock
    }

    pub fn banner(&self, what: &str) {
        println!("═══ {what} ═══");
        println!("backend : {}", self.backend.name());
        if let Some(m) = &self.model_info {
            println!("model   : {m}");
        }
        println!("memory  : {}", self.watch_desc);
        if let Some(l) = &self.label {
            println!("label   : {l}");
        }
        if self.template != Template::Raw {
            println!("template: {:?}", self.template);
        }
        if !self.tags.is_empty() {
            let t: Vec<String> = self.tags.iter().map(|(k, v)| format!("{k}={v}")).collect();
            println!("tags    : {}", t.join(" "));
        }
        if self.simulated() {
            println!(
                "!!! SIMULATED BACKEND — these numbers come from a cost model, not your hardware."
            );
            println!(
                "!!! Use them to learn the harness; never as evidence. Use --backend llama-server|ollama for real runs."
            );
        }
    }

    /// One measured inference call.
    pub fn infer(&mut self, prompt: &str, params: &GenParams) -> Result<Generation> {
        let sampler = Sampler::start(self.watch_pid, Duration::from_millis(20));
        let wrapped = self.template.apply(prompt);
        let res = self.backend.generate(&wrapped, params);
        let (start, peak) = sampler.finish();
        let mut g = res?;
        g.start_rss_bytes = start;
        g.peak_rss_bytes = peak;
        Ok(g)
    }

    pub fn record(&self, kind: Kind, group: &str, experiment: &str) -> Record {
        Record {
            id: new_id(match kind {
                Kind::Run => "run",
                Kind::Agent => "agent",
                Kind::Prediction => "pred",
            }),
            ts: now_secs(),
            kind,
            group: group.to_string(),
            lesson: self.lesson,
            experiment: experiment.to_string(),
            label: self.label.clone(),
            tags: self.tags.clone(),
            backend: self.backend.name(),
            model: self.model_info.clone(),
            params: None,
            var: None,
            value: None,
            metrics: BTreeMap::new(),
            simulated: self.simulated(),
            predictions: Vec::new(),
            text_preview: String::new(),
            warmup: false,
        }
    }

    /// Gather predictions: CLI flags win; else ask interactively; else none.
    pub fn collect_predictions(
        &self,
        default_metrics: &[&str],
        points: &[String],
    ) -> Vec<Prediction> {
        if !self.cli_predictions.is_empty() {
            return self.cli_predictions.clone();
        }
        if self.no_predict || !predict::interactive() {
            return Vec::new();
        }
        predict::ask(default_metrics, points)
    }

    /// Score predictions against per-point means and write a Prediction record.
    pub fn score_and_record(
        &self,
        group: &str,
        experiment: &str,
        preds: &[Prediction],
        points: &[String],
        means: &[BTreeMap<String, f64>],
        var: Option<&str>,
    ) -> Result<()> {
        let mut results = Vec::new();
        for p in preds {
            if p.values.len() != points.len() {
                eprintln!(
                    "warning: prediction for {} has {} values but there are {} points; skipped",
                    p.metric,
                    p.values.len(),
                    points.len()
                );
                continue;
            }
            for (i, v) in p.values.iter().enumerate() {
                if let Some(actual) = means[i].get(&p.metric) {
                    let mut r = predict::score(&p.metric, *v, *actual, self.tolerance_pct);
                    if !points[i].is_empty() {
                        r.metric = format!("{}@{}", p.metric, points[i]);
                    }
                    results.push(r);
                } else {
                    eprintln!("warning: metric {} was not measured in this run", p.metric);
                }
            }
        }
        predict::print_results(
            &results,
            &format!("(tolerance ±{:.0}%)", self.tolerance_pct),
        );
        if preds.is_empty() && !self.no_predict {
            println!(
                "\n(no prediction recorded — progress only counts predicted runs. Use --predict metric=value)"
            );
        }
        let mut rec = self.record(Kind::Prediction, group, experiment);
        rec.var = var.map(str::to_string);
        rec.predictions = results;
        if means.len() == 1 {
            rec.metrics = means[0].clone();
        }
        self.notebook.append(&rec)
    }
}

/// Print a mean/sd/cv/p50/p95 table for the given metrics over rows.
pub fn print_summary(rows: &[BTreeMap<String, f64>], names: &[&str]) {
    println!(
        "  {:<18} {:>10} {:>9} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "metric", "mean", "sd", "cv%", "p50", "p95", "min", "max"
    );
    for name in names {
        let xs: Vec<f64> = rows.iter().filter_map(|r| r.get(*name).copied()).collect();
        if let Some(s) = stats::Summary::of(&xs) {
            println!(
                "  {:<18} {:>10} {:>9} {:>6.1} {:>10} {:>10} {:>10} {:>10}",
                name,
                metrics::short(s.mean),
                metrics::short(s.sd),
                s.cv() * 100.0,
                metrics::short(s.p50),
                metrics::short(s.p95),
                metrics::short(s.min),
                metrics::short(s.max),
            );
        }
    }
}

pub fn means(rows: &[BTreeMap<String, f64>]) -> BTreeMap<String, f64> {
    let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for r in rows {
        for (k, v) in r {
            let e = acc.entry(k.clone()).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
    }
    acc.into_iter()
        .map(|(k, (s, n))| (k, s / n as f64))
        .collect()
}

pub fn one_line(i: usize, g: &Generation, warm: bool) -> String {
    let mut s = format!(
        "  {}#{:<2} prefill {:>5} tok {:>9} | decode {:>4} tok {:>9} = {:>7} | wall {:>9}",
        if warm { "warm " } else { "" },
        i,
        g.prompt_tokens,
        g.prefill_ms
            .map(|m| format!("{m:.1} ms"))
            .unwrap_or("?".into()),
        g.completion_tokens,
        g.decode_ms
            .map(|m| format!("{m:.1} ms"))
            .unwrap_or("?".into()),
        format!("{:.1} t/s", g.decode_tok_s()),
        format!("{:.1} ms", g.wall_ms),
    );
    if let Some(o) = g.overhead_ms() {
        s += &format!(" | ovh {o:.1} ms");
    }
    if let Some(l) = g.load_ms.filter(|l| *l > 1.0) {
        s += &format!(" | load {l:.1} ms");
    }
    if let (Some(a), Some(b)) = (g.start_rss_bytes, g.peak_rss_bytes) {
        s += &format!(" | rss {}→{}", memory::fmt_bytes(a), memory::fmt_bytes(b));
    }
    s
}
