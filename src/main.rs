//! machine — a predict → run → compare lab for agentic loops on local inference.

mod advisor;
mod agent;
mod autopilot;
mod backend;
mod calc;
mod env;
mod experiment;
mod lab;
mod learnings;
mod lessons;
mod memory;
mod metrics;
mod notebook;
mod predict;
mod scientist;
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

/// Which backend to talk to and how. Shared by every command that infers.
#[derive(Args, Clone)]
struct BackendArgs {
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
}

impl BackendArgs {
    fn config(&self) -> BackendConfig {
        BackendConfig {
            kind: self.backend,
            url: self.url.clone(),
            model: self.model.clone(),
        }
    }

    /// The same settings as command-line flags, so proposals are copy-pasteable.
    fn flags(&self) -> String {
        use clap::ValueEnum;
        let name = |v: Option<clap::builder::PossibleValue>| {
            v.map(|p| p.get_name().to_string()).unwrap_or_default()
        };
        let mut s = format!("--backend {} ", name(self.backend.to_possible_value()));
        if let Some(u) = &self.url {
            s += &format!("--url {u} ");
        }
        if let Some(m) = &self.model {
            s += &format!("--model {m} ");
        }
        if self.template != lab::Template::Raw {
            s += &format!("--template {} ", name(self.template.to_possible_value()));
        }
        s
    }
}

#[derive(Args, Clone)]
struct Common {
    #[command(flatten)]
    b: BackendArgs,
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
    /// Run the predict → run → score → learn cycle unattended, one single-variable trial at a time.
    Autopilot {
        #[command(flatten)]
        b: BackendArgs,
        /// Wall-clock budget in minutes.
        #[arg(long, default_value_t = 30.0)]
        budget: f64,
        /// Maximum number of trials.
        #[arg(long, default_value_t = 40)]
        max_steps: usize,
        /// Only propose experiments about this area (baseline/noise prerequisites still run).
        #[arg(long, value_enum)]
        focus: Option<autopilot::Focus>,
        /// Print the trials it would run, with predictions, then exit. Runs nothing.
        #[arg(long)]
        dry_run: bool,
        /// Ask for confirmation before every trial.
        #[arg(long)]
        require_approval: bool,
        /// Prediction tolerance in percent.
        #[arg(long, default_value_t = predict::DEFAULT_TOLERANCE_PCT)]
        tolerance: f64,
    },
    /// Fit the current cost models and effects from the notebook (no new runs).
    Analyze {
        #[command(flatten)]
        b: BackendArgs,
    },
    /// Print the explicit belief state; --what-if asks the surrogate about a configuration.
    Beliefs {
        #[command(flatten)]
        b: BackendArgs,
        /// Print beliefs.json instead of the table.
        #[arg(long)]
        json: bool,
        /// e.g. "prompt_tokens=512 max_tokens=64 cache=true"
        #[arg(long)]
        what_if: Option<String>,
    },
    /// Rank the next experiments by expected learning × value ÷ cost; --loop runs them.
    Scientist {
        #[command(flatten)]
        b: BackendArgs,
        /// Run the selected experiments (autopilot loop with the scientist's selector).
        #[arg(long = "loop")]
        run_loop: bool,
        /// Wall-clock budget in minutes (--loop).
        #[arg(long, default_value_t = 45.0)]
        budget: f64,
        #[arg(long, default_value_t = 40)]
        max_steps: usize,
        #[arg(long, value_enum)]
        focus: Option<autopilot::Focus>,
        /// With --loop: print the sequence it would run, then exit.
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        require_approval: bool,
        #[arg(long, default_value_t = predict::DEFAULT_TOLERANCE_PCT)]
        tolerance: f64,
        /// How many ranked candidates to show.
        #[arg(long, default_value_t = 8)]
        top: usize,
    },
    /// Write a markdown research note from beliefs, ranking and learnings.
    Report {
        #[command(flatten)]
        b: BackendArgs,
        /// Write to this file instead of stdout.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// What the autopilot learned, one entry per trial.
    Learnings {
        /// Print the raw JSONL records.
        #[arg(long)]
        json: bool,
        /// Show only the last N.
        #[arg(long, default_value_t = 20)]
        tail: usize,
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
        backend: c.b.config(),
        watch: c.b.watch.clone(),
        label: c.label.clone(),
        tags: parse_tags(&c.tags)?,
        lesson: c.lesson,
        tolerance_pct: c.tolerance,
        predictions: c.predict.clone(),
        no_predict: c.no_predict,
        quiet,
        template: c.b.template,
        home: None,
    })
}

/// Notebook rows and learnings for the selected backend, with a hint when
/// the notebook has data only for other backends.
fn scientific_inputs(
    b: &BackendArgs,
) -> Result<(Vec<notebook::Record>, Vec<learnings::Learning>, String)> {
    let nb = notebook::Notebook::open()?;
    let all = nb.all()?;
    let name = backend::build(&b.config())?.name();
    let records = autopilot::relevant(&all, &name);
    if records.is_empty() && !all.is_empty() {
        let others: std::collections::BTreeSet<&str> =
            all.iter().map(|r| r.backend.as_str()).collect();
        eprintln!(
            "note: no rows for {name}; the notebook has rows for: {}. Pass --backend (and --url) to select one.",
            others.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    let learned = learnings::load(&nb)?
        .into_iter()
        .filter(|l| l.backend == name)
        .collect();
    Ok((records, learned, name))
}

fn scientist_once(
    records: &[notebook::Record],
    learned: &[learnings::Learning],
    name: &str,
    opts: &autopilot::Options,
    top: usize,
) {
    use scientist::{analyze, beliefs::Status, selector};
    let a = analyze::analyze(records, learned, name);
    let count = |s: Status| a.beliefs.beliefs.iter().filter(|b| b.status == s).count();
    println!("═══ SCIENTIST ═══  {name}");
    println!(
        "beliefs: {} known, {} uncertain, {} untested  ·  {} anomalies  (details: machine analyze / machine beliefs)",
        count(Status::Known),
        count(Status::Uncertain),
        count(Status::Untested),
        a.beliefs.anomalies.len()
    );
    for an in &a.beliefs.anomalies {
        if !an.key.starts_with("prediction_miss") {
            println!("  ✗ {}", an.title);
        }
    }
    let state = autopilot::AutopilotState::new(opts.budget_min, opts.max_steps, opts.focus);
    if selector::sparse(&a) {
        println!(
            "\nData is too sparse to steer by beliefs (<{} run rows or no fitted lines).",
            selector::MIN_ROWS
        );
        if let Some(p) = autopilot::propose_next(&state, records) {
            println!("Fallback — the deterministic checklist's next trial:");
            autopilot::print_proposal(1, &p, opts);
        }
        return;
    }
    println!(
        "\n── RANKED NEXT EXPERIMENTS (score = gain × (0.25 + 0.75·value) / (1 + log2(1 + cost_s/10)))"
    );
    let ranked = selector::rank(&a, records, &state);
    selector::print_ranking(&ranked, top, &opts.backend_flags);
    if let Some(p) = autopilot::propose_next(&state, records) {
        println!(
            "\n  (the deterministic checklist would run next: {})",
            p.key
        );
    }
    println!(
        "\nRun them: machine scientist --loop --budget {:.0}",
        opts.budget_min
    );
}

fn scientist_dry_run(records: &[notebook::Record], name: &str, opts: &autopilot::Options) {
    use scientist::{analyze, selector};
    let a = analyze::analyze(records, &[], name);
    autopilot::print_header(
        records,
        name,
        opts,
        true,
        "scientist (expected information gain × value ÷ cost)",
    );
    let mut state = autopilot::AutopilotState::new(opts.budget_min, opts.max_steps, opts.focus);
    let mut n = 0;
    let stop = loop {
        let next = if selector::sparse(&a) {
            autopilot::propose_next(&state, records).map(|p| (p, None))
        } else {
            selector::rank(&a, records, &state)
                .into_iter()
                .find(|c| c.score >= selector::MIN_SCORE)
                .map(|c| {
                    let s = format!(
                        "score {:.3} = gain {:.2} × value {:.2}",
                        c.score, c.gain, c.value
                    );
                    (c.proposal, Some(s))
                })
        };
        if let Some(stop) = autopilot::gate(&state, next.as_ref().map(|n| &n.0)) {
            break stop;
        }
        let Some((p, score)) = next else {
            break autopilot::Stop::NothingLeft;
        };
        n += 1;
        autopilot::print_proposal(n, &p, opts);
        if let Some(s) = score {
            println!("    {s}");
        }
        state.tried.insert(p.key.clone());
        state.spent_ms += p.est_ms;
        state.steps += 1;
    };
    println!(
        "\n── would run {n} trial(s), ~{} of the {:.0} min budget. Stops: {stop}.",
        autopilot::fmt_ms(state.spent_ms),
        opts.budget_min
    );
    println!(
        "   (scores here use today's beliefs; the real loop refits after every trial and re-ranks)"
    );
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
        Cmd::Autopilot {
            b,
            budget,
            max_steps,
            focus,
            dry_run,
            require_approval,
            tolerance,
        } => {
            let opts = autopilot::Options {
                budget_min: budget,
                max_steps,
                focus,
                require_approval,
                tolerance_pct: tolerance,
                backend_flags: b.flags(),
            };
            let records = notebook::Notebook::open()?.all()?;
            let backend_name = backend::build(&b.config())?.name();
            if dry_run {
                autopilot::dry_run(&records, &backend_name, &opts);
            } else {
                let mut lab = lab::Lab::new(lab::LabOptions {
                    backend: b.config(),
                    watch: b.watch.clone(),
                    label: None,
                    tags: BTreeMap::new(),
                    lesson: None,
                    tolerance_pct: tolerance,
                    predictions: Vec::new(),
                    no_predict: false,
                    quiet: false,
                    template: b.template,
                    home: None,
                })?;
                let session = autopilot::execute(&mut lab, &opts)?;
                let held = session
                    .trials
                    .iter()
                    .filter(|(_, o)| !o.results.is_empty() && o.results.iter().all(|r| r.passed))
                    .count();
                println!(
                    "{} trial(s) in {}, {held} with every prediction within tolerance. Stopped: {}.",
                    session.trials.len(),
                    autopilot::fmt_ms(session.spent_ms),
                    session.stop
                );
                println!("Review: machine learnings   ·   machine progress   ·   machine wrap-up");
            }
        }
        Cmd::Analyze { b } => {
            let (records, learned, name) = scientific_inputs(&b)?;
            let a = scientist::analyze::analyze(&records, &learned, &name);
            scientist::analyze::print(&a);
            let nb = notebook::Notebook::open()?;
            if !records.is_empty() {
                let p = a.beliefs.save(&nb.dir())?;
                println!("\nbeliefs saved to {}", p.display());
            }
            let state = autopilot::AutopilotState::new(1e9, 1000, None);
            let ranked = scientist::selector::rank(&a, &records, &state);
            println!("\n── REMAINING HIGH-VALUE EXPERIMENTS");
            if scientist::selector::sparse(&a) {
                println!(
                    "  (data too sparse for the selector — `machine autopilot` will gather the basics)"
                );
            } else {
                scientist::selector::print_ranking(&ranked, 6, &b.flags());
            }
        }
        Cmd::Beliefs { b, json, what_if } => {
            let (records, learned, name) = scientific_inputs(&b)?;
            let a = scientist::analyze::analyze(&records, &learned, &name);
            if json {
                println!("{}", serde_json::to_string_pretty(&a.beliefs)?);
            } else if let Some(q) = what_if {
                let q = scientist::surrogate::CallQuery::parse(&q, a.surrogate.base_prompt_tokens)?;
                a.surrogate.print_query(&q);
            } else {
                a.beliefs.print();
            }
        }
        Cmd::Scientist {
            b,
            run_loop,
            budget,
            max_steps,
            focus,
            dry_run,
            require_approval,
            tolerance,
            top,
        } => {
            let (records, learned, name) = scientific_inputs(&b)?;
            let opts = autopilot::Options {
                budget_min: budget,
                max_steps,
                focus,
                require_approval,
                tolerance_pct: tolerance,
                backend_flags: b.flags(),
            };
            if !run_loop {
                scientist_once(&records, &learned, &name, &opts, top);
            } else if dry_run {
                scientist_dry_run(&records, &name, &opts);
            } else {
                let nb = notebook::Notebook::open()?;
                let mut lab = lab::Lab::new(lab::LabOptions {
                    backend: b.config(),
                    watch: b.watch.clone(),
                    label: None,
                    tags: BTreeMap::new(),
                    lesson: None,
                    tolerance_pct: tolerance,
                    predictions: Vec::new(),
                    no_predict: false,
                    quiet: false,
                    template: b.template,
                    home: None,
                })?;
                let mut planner = scientist::selector::Scientist::new(&name, Some(nb.dir()));
                let session = autopilot::execute_planner(&mut lab, &opts, &mut planner)?;
                println!(
                    "{} trial(s) in {}. Stopped: {}.",
                    session.trials.len(),
                    autopilot::fmt_ms(session.spent_ms),
                    session.stop
                );
                println!(
                    "Review: machine beliefs   ·   machine learnings   ·   machine report --out report.md"
                );
            }
        }
        Cmd::Report { b, out } => {
            let (records, learned, name) = scientific_inputs(&b)?;
            let a = scientist::analyze::analyze(&records, &learned, &name);
            let state = autopilot::AutopilotState::new(1e9, 1000, None);
            let ranked = scientist::selector::rank(&a, &records, &state);
            let md = scientist::report::markdown(&a, &ranked, &learned, &b.flags());
            match out {
                Some(p) => {
                    std::fs::write(&p, md)?;
                    println!("wrote {}", p.display());
                }
                None => print!("{md}"),
            }
        }
        Cmd::Learnings { json, tail } => {
            let ls = learnings::load(&notebook::Notebook::open()?)?;
            if json {
                for l in ls.iter().skip(ls.len().saturating_sub(tail)) {
                    println!("{}", serde_json::to_string(l)?);
                }
            } else {
                learnings::print(&ls, tail);
            }
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
        Cmd::WrapUp => {
            let nb = notebook::Notebook::open()?;
            advisor::wrap_up(&nb.all()?, &learnings::load(&nb)?);
        }
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
