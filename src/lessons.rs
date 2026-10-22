//! The curriculum. Each lesson has the same shape:
//! concept → experiment → prediction → expected numbers → what to look for
//! → challenge, plus check questions that point at a specific metric.

pub struct Check {
    pub question: &'static str,
    /// Where the evidence lives — shown only after the student answers.
    pub evidence: &'static str,
}

pub struct Lesson {
    pub n: u8,
    pub title: &'static str,
    pub concept: &'static str,
    pub experiment: &'static str,
    pub rust: &'static str,
    pub rust_beginner: &'static str,
    pub predict: &'static str,
    pub expect: &'static str,
    pub look_for: &'static str,
    pub challenge: &'static str,
    pub misconception: &'static str,
    pub checks: &'static [Check],
    /// Experiment-name prefixes in the notebook that count for this lesson.
    pub counts: &'static [&'static str],
}

pub const LESSONS: &[Lesson] = &[
    Lesson {
        n: 1,
        title: "Minimal inference loop",
        concept: "One completion is two phases — prefill (read the whole prompt at once) and decode (one token per step) — and each has its own cost you can measure.",
        experiment: "machine run --lesson 1 --max-tokens 64
# real model:  llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 2048 --port 8080
#              machine run --backend llama-server --lesson 1 --max-tokens 64",
        rust: "src/lab.rs  Lab::infer()      — starts the RSS sampler, calls the backend, stops the sampler
src/backend/llama_server.rs  — POST /completion, reads timings.prompt_ms / predicted_ms
src/metrics.rs from_generation() — turns one call into named numbers",
        rust_beginner: "Lab::infer takes `&mut self` because the backend keeps state (an HTTP agent, a call counter).
`Box<dyn Backend>` lets one variable hold any backend; the trait is the contract.
`?` returns the error to the caller instead of panicking — read it as 'or bail out'.",
        predict: "Before running: what decode tok/s and wall_ms do you expect for 64 tokens? Which is larger: prefill_ms or decode_ms?",
        expect: "One line per run: prefill N tok / X ms | decode 64 tok / Y ms = Z tok/s | wall W ms | ovh O ms | rss A→B.
For a short prompt, decode_ms ≫ prefill_ms, and wall ≈ prefill + decode + a few ms overhead.",
        look_for: "prefill_tok_s is usually many times decode_tok_s: prefill processes tokens in parallel, decode is sequential.
overhead_ms is the cost you can't blame on the model (HTTP + JSON). rss_delta is small: the KV cache is allocated when the server starts.",
        challenge: "Double --max-tokens to 128. Predict wall_ms first. Did it double? If not, which component didn't scale?",
        misconception: "'tok/s' alone is ambiguous: always say prefill or decode.",
        checks: &[
            Check { question: "Which phase dominated wall time, and by roughly how much?", evidence: "compare prefill_ms and decode_ms in the run line" },
            Check { question: "If you halve max_tokens, which number barely moves?", evidence: "prefill_ms — it depends on prompt length, not output length" },
            Check { question: "Where does the gap between wall_ms and prefill+decode come from?", evidence: "overhead_ms (and load_ms on Ollama's first call)" },
        ],
        counts: &["single", "repeat"],
    },
    Lesson {
        n: 2,
        title: "Measurement discipline",
        concept: "A single run is an anecdote; the same prompt run many times gives a distribution whose spread (cv%) sets the smallest effect you can detect.",
        experiment: "machine run --lesson 2 --repeat 10              # notice run #1
machine run --lesson 2 --repeat 10 --warmup 1   # does cv% drop?",
        rust: "src/stats.rs Summary::of() — mean, sd, p50/p95, cv
src/experiment.rs diagnose() — flags warm-up outliers, cache reuse, high cv",
        rust_beginner: "Summary::of returns Option<Summary>: None for empty input, so the caller must handle 'no data'.
`sort_by(|a, b| a.total_cmp(b))` sorts f64 safely (f64 isn't Ord because of NaN).",
        predict: "Predict the cv% of decode_tok_s over 10 runs, and whether run #1 will be slower than the median.",
        expect: "SUMMARY table with cv% per metric. On a quiet machine decode cv% is often a few percent; on a busy laptop it can exceed 10%.",
        look_for: "The 'WHAT THE NUMBERS ARE TELLING YOU' block: warm-up outlier, cache reuse (prompt_tokens drops), noise floor.
With --temperature 0, 'distinct outputs' should be 1; if not, the backend is nondeterministic.",
        challenge: "Define your success signal in one line: e.g. 'decode ≥ X tok/s with cv < 5% and output identical across runs'. Re-run until it holds or you know why it doesn't.",
        misconception: "'It feels faster' is not evidence — only a difference larger than about 2× the combined standard error is.",
        checks: &[
            Check { question: "What is the smallest speedup you could trust on this machine right now?", evidence: "≈ 2 × cv% of decode_tok_s in the SUMMARY table" },
            Check { question: "Why might run #1 differ from the rest?", evidence: "the run #1 wall vs median note (mmap page-in, load, caches)" },
        ],
        counts: &["repeat"],
    },
    Lesson {
        n: 3,
        title: "Single-variable experiments",
        concept: "Change exactly one thing against a fixed baseline, interleave the runs, and read the slope — that slope is what lets you predict the next point.",
        experiment: "machine sweep --lesson 3 --var max-tokens --values 16,32,64,128 --repeat 3
machine sweep --lesson 3 --var prompt-tokens --values 64,256,1024 --repeat 3 --metric prefill_ms
machine sweep --lesson 3 --var temperature --values 0,0.8 --repeat 5 --metric decode_tok_s
# load-time knobs (quant, -t threads, -ngl): restart the server, tag, then compare
machine run --backend llama-server --repeat 5 --warmup 1 --label q4 --tag quant=q4_k_m
machine run --backend llama-server --repeat 5 --warmup 1 --label q8 --tag quant=q8_0
machine compare --label q4 --label q8",
        rust: "src/experiment.rs sweep() — builds every point up front, then round-robins so drift spreads evenly
slope_note() — least-squares line metric ≈ a + b·x with R²
compare() — warns when two labels differ in more than one setting",
        rust_beginner: "`match o.var { ... }` must cover every enum variant — the compiler enforces that each sweep variable is handled.
`v.parse()?` turns the string '64' into whatever type the field needs (u32, f32, bool).",
        predict: "Using lesson 1's ms/token, predict wall_ms at each max_tokens value. For prompt-tokens, predict prefill_ms at each size.",
        expect: "A SWEEP RESULT table with mean, sd and a verdict per point, plus a linear fit. max_tokens vs wall_ms should be close to a straight line (R² near 1).",
        look_for: "The slope b is your per-token decode cost in ms; the intercept a is prefill + overhead.
If R² drops at large prompt sizes, attention cost per token is growing with context. A lower-bit quant usually trades quality for speed AND memory — compare peak_rss_mib.",
        challenge: "Predict the next point off the fitted line (e.g. max_tokens=256) before running it. Within 20%?",
        misconception: "Bigger context is not free: prefill grows with prompt length, and decode slows as the KV cache grows.",
        checks: &[
            Check { question: "What is your machine's per-output-token cost in ms?", evidence: "slope b of the max_tokens → wall_ms fit" },
            Check { question: "Did temperature change decode speed in a way you could detect?", evidence: "the verdict column: 'within noise' or 'real difference'" },
            Check { question: "If q8 is slower than q4, which resource is the bottleneck?", evidence: "decode_tok_s vs peak_rss_mib in `compare`: more bytes per weight → more memory traffic per token" },
        ],
        counts: &["sweep:"],
    },
    Lesson {
        n: 4,
        title: "A simple agentic wrapper",
        concept: "An agent is a loop — observation → inference → action → new observation — so its cost is the sum of every call, and the prompt grows each step.",
        experiment: "machine agent --lesson 4 --verbose
machine agent --lesson 4 --memory-window 0   # the model never sees the tool result",
        rust: "src/agent.rs run_episode() — the loop; every call goes through Lab::infer and is added to the episode
actor_prompt() — instructions + task + history (cut by --memory-window)
src/calc.rs — the one tool: exact integer arithmetic",
        rust_beginner: "parse_action returns an enum Action { Calc, Answer, Invalid } — the match in run_episode has to handle each, so a malformed reply can't be silently ignored.
The `measure` closure captures nothing mutable from outside; it takes &mut Episode explicitly to keep the borrow checker happy.",
        predict: "Predict calls per task and wall_ms per task from your lesson-1 per-call latency. With --memory-window 0, predict accuracy.",
        expect: "One line per task: ✓/✗, answer vs truth, calls, tokens, wall. With --verbose, each call shows prompt+completion tokens and the parsed action.",
        look_for: "prompt_tokens grows on every step (history is appended). Two calls per correct task is the ideal; format_errors cost a whole call each.
With --memory-window 0 the model can't see the calculator result, so it loops until max_steps.",
        challenge: "Predict end-to-end wall for a task as (calls × per-call latency). Where is the prediction off, and which call was the surprise?",
        misconception: "More calls can make the system slower AND less reliable — each call is another chance for a format error.",
        checks: &[
            Check { question: "Why is the second actor call more expensive in prefill than the first?", evidence: "prompt_tokens in the --verbose trace grows by the length of the history line" },
            Check { question: "What happens to accuracy and calls with --memory-window 0, and why?", evidence: "the loop never sees '-> <result>', so it repeats CALC until --max-steps" },
        ],
        counts: &["agent:"],
    },
    Lesson {
        n: 5,
        title: "Critic / verifier",
        concept: "A critic is one more inference call per answer; it's worth it only if the accuracy it buys exceeds the latency and tokens it costs.",
        experiment: "machine agent --lesson 5 --critic none   --temperature 0.8 --repeat 2 --label nocritic
machine agent --lesson 5 --critic model  --temperature 0.8 --repeat 2 --label critic
machine agent --lesson 5 --critic oracle --temperature 0.8 --repeat 2 --label oracle
machine compare --label nocritic --label critic --label oracle --metric success --metric calls --metric wall_ms",
        rust: "src/agent.rs CriticMode — None | Model | Oracle; critic_prompt() asks YES/NO with max_tokens=3
the critic's false accepts / false rejects are counted against ground truth",
        rust_beginner: "CriticMode derives clap::ValueEnum, so `--critic model` parses straight into the enum.",
        predict: "Predict the accuracy gain from the model critic and its share of total time. Will the oracle beat it? By how much?",
        expect: "critic_calls ≈ number of answers; critic share of time printed; false accepts/rejects counted.",
        look_for: "The oracle is the ceiling: (oracle − none) is the most any critic could recover; (critic − none) is what yours recovered.
A small model as its own critic often agrees with itself — watch false accepts. Critic calls are cheap in decode (≤3 tokens) but pay full prefill.",
        challenge: "Compute tokens per correct answer for each mode. Which mode is cheapest per correct answer, not per call?",
        misconception: "A critic that 'looks smart' in one transcript proves nothing — count its false accepts over many episodes.",
        checks: &[
            Check { question: "Where does the critic's time go: prefill or decode?", evidence: "--verbose critic lines: large prompt tokens, ≤3 completion tokens" },
            Check { question: "What's the cost per extra correct answer the critic bought?", evidence: "Δwall_ms ÷ Δcorrect between the critic and nocritic labels" },
        ],
        counts: &["agent:critic=model", "agent:critic=oracle"],
    },
    Lesson {
        n: 6,
        title: "Closing the optimization loop",
        concept: "Use the notebook — not intuition — to choose the next single change: the metric that's furthest from your success signal, the knob with the largest measured slope.",
        experiment: "machine progress     # which predictions you got right
machine next         # the advisor reads the notebook and proposes an untried experiment
machine sweep --var max-tokens ... or machine agent --memory-window 2 ...",
        rust: "src/advisor.rs — rules over the notebook: noise first, then untried variables, then agent structure",
        rust_beginner: "The notebook is plain JSONL — `jq` it or open it; nothing is hidden.",
        predict: "Before running `machine next`, write down what YOU think the next experiment should be and why.",
        expect: "A suggestion with a reason tied to a recorded number (e.g. 'cv 9% on decode — fix noise first').",
        look_for: "Whether your own pick and the advisor's agree. If not, which recorded number is each of you reasoning from?",
        challenge: "Design and run one experiment the advisor did not suggest. Write the prediction first.",
        misconception: "Global rewrites (new backend, new prompt, new model at once) destroy causal information — keep it to one change per comparison.",
        checks: &[
            Check { question: "Which single component would you remove to make the agent faster, and what would it cost in accuracy?", evidence: "compare critic vs nocritic: Δwall and Δsuccess" },
            Check { question: "Can you predict the next experiment's primary metric within 20%?", evidence: "machine progress — your recent prediction errors" },
        ],
        counts: &[],
    },
];

pub fn get(n: u8) -> Option<&'static Lesson> {
    LESSONS.iter().find(|l| l.n == n)
}

pub fn print(l: &Lesson, beginner: bool, reveal: bool) {
    println!("═══ LESSON {} — {} ═══\n", l.n, l.title);
    println!("CONCEPT\n  {}\n", l.concept);
    println!("EXPERIMENT");
    for line in l.experiment.lines() {
        println!("  {line}");
    }
    println!("\nWHERE IT LIVES IN THE CODE");
    for line in l.rust.lines() {
        println!("  {line}");
    }
    if beginner {
        println!("\nRUST NOTES (shown because your profile says you're new to Rust)");
        for line in l.rust_beginner.lines() {
            println!("  {line}");
        }
    }
    println!("\nPREDICT FIRST\n  {}\n", l.predict);
    println!("WHAT THE OUTPUT SHOULD LOOK LIKE");
    for line in l.expect.lines() {
        println!("  {line}");
    }
    println!("\nWHAT TO LOOK FOR");
    for line in l.look_for.lines() {
        println!("  {line}");
    }
    println!("\nMISCONCEPTION TO AVOID\n  {}", l.misconception);
    println!("\nCHALLENGE\n  {}", l.challenge);
    println!(
        "\nCHECK YOURSELF (answer before `machine lesson {} --reveal`)",
        l.n
    );
    for (i, c) in l.checks.iter().enumerate() {
        println!("  {}. {}", i + 1, c.question);
        if reveal {
            println!("     evidence → {}", c.evidence);
        }
    }
}
