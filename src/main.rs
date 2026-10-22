//! machine — a predict → run → compare lab for agentic loops on local inference.

mod advisor;
mod agent;
#[allow(dead_code)] // wired into the CLI in the next step
mod autopilot;
mod backend;
mod calc;
mod env;
mod experiment;
mod lab;
mod lessons;
mod memory;
mod metrics;
mod notebook;
mod predict;
mod stats;

use anyhow::{Result, bail};
use backend::{BackendConfig, BackendKind, GenParams};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;

#[derive(Parser)]
#[command(
    name = "machine",
    version,
    about = "Predict → run → compare: learn to measure and optimize agentic loops on local inference.",
    long_about = "Every experiment asks for your prediction first, runs against a local model\n(llama-server, Ollama, or a clearly-labelled simulator), records raw numbers to\n.machine/notebook.jsonl, and scores your prediction.\n\nStart with:  machine start   then   machine lesson 1"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// Inference backend.
    #[arg(long, value_enum, default_value = "mock", env = "MACHINE_BACKEND")]
    backend: BackendKind,
    /// Backend base URL (default: http://127.0.0.1:8080 llama-server, :11434 ollama).
    #[arg(long, env = "MACHINE_URL")]
    url: Option<String>,
    /// Model name (Ollama only, e.g. qwen2.5:0.5b).
    #[arg(long, env = "MACHINE_MODEL")]
    model: Option<String>,
    /// Prompt wrapping. Use chatml for Qwen-style instruct models.
    #[arg(long, value_enum, default_value = "raw", env = "MACHINE_TEMPLATE")]
    template: lab::Template,
    /// Process to sample memory from: auto | self | none | pid:<N> | name:<substr>.
    #[arg(long, default_value = "auto")]
    watch: String,
    /// Label for this group of runs (used by `compare`).
    #[arg(long)]
    label: Option<String>,
    /// key=value tags for load-time settings, e.g. --tag quant=q4_k_m --tag threads=8.
    #[arg(long = "tag")]
    tags: Vec<String>,
    /// Lesson this run counts toward.
    #[arg(long)]
    lesson: Option<u8>,
    /// Prediction(s): metric=value, or metric=v1,v2,... for sweeps.
    #[arg(long = "predict")]
    predict: Vec<String>,
    /// Don't ask for predictions.
    #[arg(long)]
    no_predict: bool,
    /// Prediction tolerance in percent.
    #[arg(long, default_value_t = predict::DEFAULT_TOLERANCE_PCT)]
    tolerance: f64,
}

#[derive(Args, Clone)]
struct Gen {
    #[arg(long, default_value_t = 64)]
    max_tokens: u32,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Let llama-server reuse the KV cache for a shared prompt prefix.
    #[arg(long)]
    cache_prompt: bool,
    /// Keep generating past end-of-sequence (llama-server) so output length = --max-tokens.
    #[arg(long)]
    ignore_eos: bool,
}

impl Gen {
    fn params(&self) -> GenParams {
        GenParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            seed: Some(self.seed),
            cache_prompt: self.cache_prompt,
            stop: Vec::new(),
            ignore_eos: self.ignore_eos,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Tell the lab what you already know; adapts lesson depth.
    Start,
    /// Show a lesson card (default: your current lesson).
    Lesson {
        n: Option<u8>,
        /// Show where the evidence for each check question lives.
        #[arg(long)]
        reveal: bool,
    },
    /// One completion (or --repeat N) with full timing + memory breakdown.
    Run {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen_: Gen,
        #[arg(long, default_value = experiment::DEFAULT_PROMPT)]
        prompt: String,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// Runs to discard before measuring.
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        /// Pad the prompt to roughly this many tokens.
        #[arg(long)]
        pad_to: Option<usize>,
        #[arg(long)]
        show_text: bool,
    },
    /// Change exactly one variable across values; interleaved repeats.
    Sweep {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen_: Gen,
        #[arg(long, value_enum)]
        var: experiment::SweepVar,
        /// Comma-separated values, e.g. 16,32,64.
        #[arg(long, value_delimiter = ',', required = true)]
        values: Vec<String>,
        #[arg(long, default_value = experiment::DEFAULT_PROMPT)]
        prompt: String,
        #[arg(long, default_value_t = 3)]
        repeat: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Metric to tabulate and predict (default depends on --var).
        #[arg(long)]
        metric: Option<String>,
        #[arg(long)]
        pad_to: Option<usize>,
        /// Only print the result table.
        #[arg(long)]
        quiet: bool,
    },
    /// Agent loop: observation → inference → tool action, with optional critic.
    Agent {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen_: Gen,
        /// Task indices (default: all). See `machine tasks`.
        #[arg(long, value_delimiter = ',')]
        tasks: Vec<usize>,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        #[arg(long, default_value_t = 6)]
        max_steps: usize,
        #[arg(long, value_enum, default_value = "none")]
        critic: agent::CriticMode,
        /// History lines the model sees (default: all).
        #[arg(long)]
        memory_window: Option<usize>,
        /// Print every inference call.
        #[arg(long)]
        verbose: bool,
    },
    /// Side-by-side comparison of labelled runs.
    Compare {
        #[arg(long = "label", required = true)]
        labels: Vec<String>,
        #[arg(long = "metric")]
        metrics: Vec<String>,
    },
    /// Which lessons you've passed (by prediction accuracy).
    Progress,
    /// Suggest the next single-variable experiment from your notebook.
    Next,
    /// End-of-session check, weakest idea, and WHAT SHOULD I BUILD NEXT?
    WrapUp,
    /// Hardware and backend probe.
    Env {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        llama_url: String,
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        ollama_url: String,
    },
    /// List metric names (for --predict and --metric).
    Metrics,
    /// List the agent's tasks.
    Tasks,
    /// Show the notebook location and the last N rows.
    Notebook {
        #[arg(long, default_value_t = 10)]
        tail: usize,
    },
}

fn parse_tags(tags: &[String]) -> Result<BTreeMap<String, String>> {
    tags.iter()
        .map(|t| match t.split_once('=') {
            Some((k, v)) => Ok((k.trim().to_string(), v.trim().to_string())),
            None => bail!("--tag must be key=value, got {t:?}"),
        })
        .collect()
}

fn make_lab(c: &Common, quiet: bool) -> Result<lab::Lab> {
    lab::Lab::new(lab::LabOptions {
        backend: BackendConfig {
            kind: c.backend,
            url: c.url.clone(),
            model: c.model.clone(),
        },
        watch: c.watch.clone(),
        label: c.label.clone(),
        tags: parse_tags(&c.tags)?,
        lesson: c.lesson,
        tolerance_pct: c.tolerance,
        predictions: c.predict.clone(),
        no_predict: c.no_predict,
        quiet,
        template: c.template,
    })
}

fn level(q: &str) -> u8 {
    let a = predict::ask_line(&format!("  {q} [0 none / 1 some / 2 comfortable]: "))
        .unwrap_or_default();
    a.parse::<u8>().unwrap_or(0).min(2)
}

fn start() -> Result<()> {
    println!("═══ WHAT DO YOU ALREADY KNOW? ═══");
    if !predict::interactive() {
        println!("(non-interactive: run this in a terminal)");
        return Ok(());
    }
    let mut p = notebook::Profile {
        rust: level("Rust in general"),
        ownership: level("ownership & borrowing"),
        async_rust: level("async Rust"),
        cli: level("building CLIs"),
        local_inference: level("local inference (llama.cpp, Candle, Ollama…)"),
        ..Default::default()
    };
    p.tools_used = predict::ask_line("  which tools have you used? (comma list, blank for none): ")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    p.hardware_notes =
        predict::ask_line("  your hardware in one line (e.g. 'M2 16GB', 'Ryzen 7 + RTX 3060'): ")
            .unwrap_or_default();
    p.save()?;
    println!("\nsaved to {}", notebook::Profile::path().display());
    if p.beginner_rust() {
        println!(
            "Lesson cards will include Rust notes. Read src/lab.rs first — it's the hot path."
        );
    }
    if p.local_inference == 0 {
        println!(
            "New to local inference: start on the simulator (default) to learn the harness, then:"
        );
        println!("  llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 2048 --port 8080");
        println!("  machine run --backend llama-server --template chatml");
    } else {
        println!(
            "You know local inference: go straight to a real backend, e.g. --backend llama-server."
        );
    }
    println!("Next: machine env   then   machine lesson 1");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Start => start()?,
        Cmd::Lesson { n, reveal } => {
            let records = notebook::Notebook::open()?.all()?;
            let n = n.unwrap_or_else(|| advisor::current_lesson(&records));
            let Some(l) = lessons::get(n) else {
                bail!("lessons are 1–{}", lessons::LESSONS.len())
            };
            let beginner = notebook::Profile::load()
                .map(|p| p.beginner_rust())
                .unwrap_or(false);
            lessons::print(l, beginner, reveal);
        }
        Cmd::Run {
            common,
            gen_,
            prompt,
            repeat,
            warmup,
            pad_to,
            show_text,
        } => {
            let mut lab = make_lab(&common, false)?;
            experiment::run(
                &mut lab,
                experiment::RunOpts {
                    prompt,
                    params: gen_.params(),
                    repeat,
                    warmup,
                    pad_to,
                    show_text,
                },
            )?;
        }
        Cmd::Sweep {
            common,
            gen_,
            var,
            values,
            prompt,
            repeat,
            warmup,
            metric,
            pad_to,
            quiet,
        } => {
            let mut lab = make_lab(&common, quiet)?;
            experiment::sweep(
                &mut lab,
                experiment::SweepOpts {
                    var,
                    values,
                    metric,
                    base: experiment::RunOpts {
                        prompt,
                        params: gen_.params(),
                        repeat,
                        warmup,
                        pad_to,
                        show_text: false,
                    },
                },
            )?;
        }
        Cmd::Agent {
            common,
            gen_,
            tasks,
            repeat,
            max_steps,
            critic,
            memory_window,
            verbose,
        } => {
            let tasks = if tasks.is_empty() {
                (0..agent::TASKS.len()).collect()
            } else {
                tasks
            };
            if let Some(bad) = tasks.iter().find(|t| **t >= agent::TASKS.len()) {
                bail!("task {bad} doesn't exist; see `machine tasks`");
            }
            let mut lab = make_lab(&common, false)?;
            let mut params = gen_.params();
            if gen_.max_tokens == 64 {
                params.max_tokens = 32; // one line is enough for the agent protocol
            }
            agent::run(
                &mut lab,
                agent::AgentOpts {
                    tasks,
                    repeat,
                    max_steps,
                    critic,
                    memory_window,
                    params,
                    verbose,
                },
            )?;
        }
        Cmd::Compare { labels, metrics } => {
            let records = notebook::Notebook::open()?.all()?;
            experiment::compare(&records, &labels, &metrics)?;
        }
        Cmd::Progress => advisor::print_progress(&notebook::Notebook::open()?.all()?),
        Cmd::Next => {
            let (why, cmd) = advisor::next(&notebook::Notebook::open()?.all()?);
            println!("WHAT SHOULD I BUILD NEXT?\n  {why}");
            for l in cmd.lines() {
                println!("  $ {l}");
            }
            println!("  Predict its primary metric before you run it.");
        }
        Cmd::WrapUp => advisor::wrap_up(&notebook::Notebook::open()?.all()?),
        Cmd::Env {
            llama_url,
            ollama_url,
        } => env::print(&llama_url, &ollama_url),
        Cmd::Metrics => {
            println!("per-call metrics (run, sweep):");
            for (n, u, d) in metrics::RUN_METRICS {
                println!("  {n:<20} {u:<6} {d}");
            }
            println!("per-episode metrics (agent):");
            for (n, u, d) in metrics::AGENT_METRICS {
                println!("  {n:<20} {u:<6} {d}");
            }
        }
        Cmd::Tasks => {
            for (i, t) in agent::TASKS.iter().enumerate() {
                println!("  {i:>2}. {}  (answer {})", t.question, t.answer);
            }
        }
        Cmd::Notebook { tail } => {
            let nb = notebook::Notebook::open()?;
            let all = nb.all()?;
            println!("{}  ({} rows)", nb.path().display(), all.len());
            for r in all.iter().rev().take(tail).rev() {
                let key: Vec<String> = ["decode_tok_s", "wall_ms", "success"]
                    .iter()
                    .filter_map(|k| {
                        r.metrics
                            .get(*k)
                            .map(|v| format!("{k}={}", metrics::short(*v)))
                    })
                    .collect();
                println!(
                    "  {:?} {:<28} {:<10} {}{}",
                    r.kind,
                    r.experiment,
                    r.label.clone().unwrap_or_default(),
                    key.join(" "),
                    if r.simulated { " [sim]" } else { "" }
                );
            }
        }
    }
    Ok(())
}
