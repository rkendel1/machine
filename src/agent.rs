//! Lessons 4–6: an agentic wrapper around local inference.
//!
//!   observation → inference (decide) → action (tool) → new observation
//!
//! The environment is a fixed set of arithmetic word problems with known
//! answers, and one tool (an exact calculator). Known answers mean every
//! episode has an unambiguous success signal, so the cost of extra calls
//! (critic, retries, longer memory) can be weighed against measured accuracy.

use crate::backend::GenParams;
use crate::lab::{self, Lab};
use crate::notebook::{Kind, new_id};
use crate::{calc, metrics, stats};
use anyhow::Result;
use std::collections::BTreeMap;
use std::time::Instant;

pub struct Task {
    pub question: &'static str,
    pub expr: &'static str,
    pub answer: i64,
}

pub const TASKS: &[Task] = &[
    Task {
        question: "A crate holds 12 boxes and each box holds 24 bolts. How many bolts are in 7 crates?",
        expr: "7*12*24",
        answer: 2016,
    },
    Task {
        question: "A server handles 340 requests per minute. How many requests does it handle in 3 hours?",
        expr: "340*60*3",
        answer: 61200,
    },
    Task {
        question: "You have 1000 dollars, spend 237 on rent and 158 on food. How much is left?",
        expr: "1000-237-158",
        answer: 605,
    },
    Task {
        question: "A model generates 45 tokens per second. How many tokens does it generate in 8 minutes?",
        expr: "45*60*8",
        answer: 21600,
    },
    Task {
        question: "A train travels 96 km per hour for 5 hours, then 72 km per hour for 3 hours. How far does it travel?",
        expr: "96*5+72*3",
        answer: 696,
    },
    Task {
        question: "936 apples are split equally into 18 bags. How many apples per bag?",
        expr: "936/18",
        answer: 52,
    },
    Task {
        question: "A rack has 4 shelves with 9 servers each, and each server has 64 GB of RAM. How many GB in total?",
        expr: "4*9*64",
        answer: 2304,
    },
    Task {
        question: "Tickets cost 17 dollars for adults and 9 dollars for children. What do 23 adults and 41 children pay?",
        expr: "17*23+9*41",
        answer: 760,
    },
    Task {
        question: "A batch job processes 125 files per run and runs 6 times a day. How many files in 2 weeks?",
        expr: "125*6*14",
        answer: 10500,
    },
    Task {
        question: "A 4096-token context is split into chunks of 256 tokens. How many chunks, minus the 3 reserved ones?",
        expr: "4096/256-3",
        answer: 13,
    },
];

fn task_in(prompt: &str) -> Option<&'static Task> {
    let after = prompt.rsplit("TASK:").next()?;
    let line = after.lines().next()?.trim();
    TASKS.iter().find(|t| t.question == line)
}

/// Used by the mock backend to simulate a model that "understands" the task.
pub fn lookup_task(prompt: &str) -> Option<&'static Task> {
    task_in(prompt)
}
pub fn lookup_answer(prompt: &str) -> Option<i64> {
    task_in(prompt).map(|t| t.answer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CriticMode {
    /// No verification: first ANSWER is final.
    None,
    /// Ask the same local model YES/NO before accepting an answer (costs inference).
    Model,
    /// Ground-truth check (free; an upper bound on what any critic could achieve).
    Oracle,
}

pub struct AgentOpts {
    pub tasks: Vec<usize>,
    pub repeat: usize,
    pub max_steps: usize,
    pub critic: CriticMode,
    /// How many history lines the model sees (None = all). Short-term memory.
    pub memory_window: Option<usize>,
    pub params: GenParams,
    pub verbose: bool,
}

const INSTRUCTIONS: &str = "You solve arithmetic word problems with a calculator tool.
Reply with exactly ONE line in one of these two forms:
CALC: <expression using numbers, + - * / and parentheses>
ANSWER: <integer>
First use CALC, then reply ANSWER with the calculator result.

Example problem: A shelf has 4 rows of 6 books. How many books?
Example first reply -> CALC: 4*6
Example tool result -> 24
Example second reply -> ANSWER: 24
";

pub fn actor_prompt(task: &Task, history: &[String], window: Option<usize>) -> String {
    let start = window.map(|w| history.len().saturating_sub(w)).unwrap_or(0);
    let mut p = format!("{INSTRUCTIONS}\nTASK: {}\nHISTORY:\n", task.question);
    if start > 0 {
        p += &format!("({start} earlier lines dropped by memory window)\n");
    }
    for h in &history[start..] {
        p += h;
        p += "\n";
    }
    p += "NEXT REPLY:";
    p
}

pub fn critic_prompt(task: &Task, history: &[String], proposed: i64) -> String {
    format!(
        "You are checking another model's work. Reply with exactly one word: YES if the answer is correct, NO if it is wrong.\nTASK: {}\nWORK:\n{}\nPROPOSED ANSWER: {proposed}\nIs the proposed answer correct?",
        task.question,
        history.join("\n")
    )
}

#[derive(Debug, PartialEq)]
pub enum Action {
    Calc(String),
    Answer(i64),
    Invalid(String),
}

pub fn parse_action(text: &str) -> Action {
    // Only the first non-empty line counts: the protocol is one line per reply.
    let Some(line) = text.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return Action::Invalid(String::new());
    };
    let upper = line.to_ascii_uppercase();
    if let Some(i) = upper.find("CALC:") {
        return Action::Calc(line[i + 5..].trim().trim_end_matches('.').to_string());
    }
    if let Some(i) = upper.find("ANSWER:") {
        let raw = line[i + 7..].trim().replace(',', "");
        let num: String = raw
            .chars()
            .skip_while(|c| !c.is_ascii_digit() && *c != '-')
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        return match num.parse() {
            Ok(n) => Action::Answer(n),
            Err(_) => Action::Invalid(line.to_string()),
        };
    }
    Action::Invalid(line.chars().take(80).collect())
}

#[derive(Default)]
struct Episode {
    m: BTreeMap<String, f64>,
    trace: Vec<String>,
    final_answer: Option<i64>,
}

impl Episode {
    fn add(&mut self, k: &str, v: f64) {
        *self.m.entry(k.to_string()).or_insert(0.0) += v;
    }
    fn max(&mut self, k: &str, v: f64) {
        let e = self.m.entry(k.to_string()).or_insert(0.0);
        *e = e.max(v);
    }
}

fn run_episode(lab: &mut Lab, task: &Task, o: &AgentOpts) -> Result<Episode> {
    let mut ep = Episode::default();
    for k in [
        "steps",
        "calls",
        "actor_calls",
        "critic_calls",
        "format_errors",
        "critic_ms",
        "critic_false_reject",
        "critic_false_accept",
    ] {
        ep.m.insert(k.into(), 0.0);
    }
    let mut history: Vec<String> = Vec::new();
    let mut actor = o.params.clone();
    if actor.stop.is_empty() {
        actor.stop = vec!["\n".into()];
    }
    let mut critic_params = o.params.clone();
    critic_params.max_tokens = 3;
    critic_params.temperature = 0.0;
    critic_params.stop = vec!["\n".into()];

    let t0 = Instant::now();
    let measure = |ep: &mut Episode, g: &crate::backend::Generation, role: &str| {
        ep.add("calls", 1.0);
        ep.add(&format!("{role}_calls"), 1.0);
        ep.add("prompt_tokens", g.prompt_tokens as f64);
        ep.add("completion_tokens", g.completion_tokens as f64);
        ep.add(
            "total_tokens",
            (g.prompt_tokens + g.completion_tokens) as f64,
        );
        ep.add("infer_ms", g.wall_ms);
        if role == "critic" {
            ep.add("critic_ms", g.wall_ms);
        }
        if let Some(p) = g.peak_rss_bytes {
            ep.max("peak_rss_mib", p as f64 / metrics::MIB);
        }
    };

    for _step in 0..o.max_steps {
        ep.add("steps", 1.0);
        let prompt = actor_prompt(task, &history, o.memory_window);
        let g = lab.infer(&prompt, &actor)?;
        measure(&mut ep, &g, "actor");
        let action = parse_action(&g.text);
        ep.trace.push(format!(
            "    actor  {:>3}+{:<2} tok {:>8.1} ms  → {:?}",
            g.prompt_tokens, g.completion_tokens, g.wall_ms, action
        ));
        match action {
            Action::Calc(expr) => match calc::eval(&expr) {
                Ok(v) => history.push(format!("CALC: {expr} -> {v}")),
                Err(e) => history.push(format!("CALC: {expr} -> ERROR: {e}")),
            },
            Action::Invalid(t) => {
                ep.add("format_errors", 1.0);
                history.push(format!(
                    "INVALID REPLY: {t} -> ERROR: reply with CALC: or ANSWER:"
                ));
            }
            Action::Answer(n) => {
                let correct = n == task.answer;
                let accepted = match o.critic {
                    CriticMode::None => true,
                    CriticMode::Oracle => correct,
                    CriticMode::Model => {
                        let cg = lab.infer(&critic_prompt(task, &history, n), &critic_params)?;
                        measure(&mut ep, &cg, "critic");
                        let yes = cg.text.trim().to_ascii_uppercase().starts_with("YES");
                        ep.trace.push(format!(
                            "    critic {:>3}+{:<2} tok {:>8.1} ms  → {:?} ({})",
                            cg.prompt_tokens,
                            cg.completion_tokens,
                            cg.wall_ms,
                            cg.text.trim(),
                            if yes == correct { "right" } else { "WRONG" }
                        ));
                        if yes && !correct {
                            ep.add("critic_false_accept", 1.0);
                        }
                        if !yes && correct {
                            ep.add("critic_false_reject", 1.0);
                        }
                        yes
                    }
                };
                ep.final_answer = Some(n);
                if accepted {
                    break;
                }
                history.push(format!(
                    "ANSWER: {n} -> CRITIC: rejected, recheck the calculation"
                ));
            }
        }
    }
    let ok = ep.final_answer == Some(task.answer);
    ep.m.insert("success".into(), if ok { 1.0 } else { 0.0 });
    ep.m.insert("wall_ms".into(), t0.elapsed().as_secs_f64() * 1000.0);
    Ok(ep)
}

pub fn run(lab: &mut Lab, o: AgentOpts) -> Result<lab::Outcome> {
    let window = o
        .memory_window
        .map(|w| w.to_string())
        .unwrap_or("all".into());
    lab.banner(&format!(
        "AGENT  critic={:?}  memory_window={window}  max_steps={}  tasks={}×{}",
        o.critic,
        o.max_steps,
        o.tasks.len(),
        o.repeat
    ));
    println!(
        "loop: observation → inference → action(CALC tool | ANSWER){}",
        if o.critic == CriticMode::None {
            ""
        } else {
            " → critic → accept/retry"
        }
    );
    println!(
        "actor params: max_tokens={} temperature={} seed={:?}",
        o.params.max_tokens, o.params.temperature, o.params.seed
    );
    let preds = lab.collect_predictions(&["success", "calls", "wall_ms"], &[String::new()]);

    let group = new_id("grp");
    let experiment = format!("agent:critic={:?}:window={window}", o.critic).to_lowercase();
    let mut rows = Vec::new();
    for r in 0..o.repeat {
        for &ti in &o.tasks {
            let task = &TASKS[ti];
            let ep = run_episode(lab, task, &o)?;
            println!(
                "  [{}] task {:>2} r{} answer {:>7} (truth {:>6})  calls {:>2}  tokens {:>5}  wall {:>9}",
                if ep.m["success"] > 0.0 { "✓" } else { "✗" },
                ti,
                r + 1,
                ep.final_answer
                    .map(|a| a.to_string())
                    .unwrap_or("none".into()),
                task.answer,
                ep.m["calls"],
                ep.m.get("total_tokens").copied().unwrap_or(0.0),
                metrics::fmt("wall_ms", ep.m["wall_ms"]),
            );
            if o.verbose {
                for t in &ep.trace {
                    println!("{t}");
                }
            }
            let mut rec = lab.record(Kind::Agent, &group, &experiment);
            rec.params = Some(o.params.clone());
            rec.var = Some("agent".into());
            rec.value = Some(format!("task{ti}"));
            rec.tags
                .insert("critic".into(), format!("{:?}", o.critic).to_lowercase());
            rec.tags.insert("memory_window".into(), window.clone());
            rec.metrics = ep.m.clone();
            lab.notebook.append(&rec)?;
            rows.push(ep.m);
        }
    }

    println!(
        "\n── AGENT SUMMARY ({} episodes) ────────────────────────────────",
        rows.len()
    );
    lab::print_summary(
        &rows,
        &[
            "success",
            "steps",
            "calls",
            "critic_calls",
            "format_errors",
            "total_tokens",
            "wall_ms",
            "critic_ms",
            "peak_rss_mib",
        ],
    );
    let col = |k: &str| -> Vec<f64> { rows.iter().filter_map(|r| r.get(k).copied()).collect() };
    let acc = stats::Summary::of(&col("success"))
        .map(|s| s.mean)
        .unwrap_or(0.0);
    let tokens: f64 = col("total_tokens").iter().sum();
    let wall: f64 = col("wall_ms").iter().sum();
    let critic_ms: f64 = col("critic_ms").iter().sum();
    let correct = acc * rows.len() as f64;
    println!(
        "\n  accuracy            {:.0}%  ({correct:.0}/{})",
        acc * 100.0,
        rows.len()
    );
    if correct > 0.0 {
        println!("  tokens per correct  {:.0}", tokens / correct);
        println!("  ms per correct      {:.0}", wall / correct);
    }
    if o.critic == CriticMode::Model {
        let fa: f64 = col("critic_false_accept").iter().sum();
        let fr: f64 = col("critic_false_reject").iter().sum();
        println!(
            "  critic share of time {:.1}%   false accepts {fa:.0}   false rejects {fr:.0}",
            100.0 * critic_ms / wall.max(1e-9)
        );
        println!(
            "  → compare against `--critic none` and `--critic oracle` with the same tasks: is the accuracy gain worth the extra calls?"
        );
    }
    let mean = lab::means(&rows);
    let points = vec![String::new()];
    let results = lab.score_and_record(
        &group,
        &experiment,
        &preds,
        &points,
        std::slice::from_ref(&mean),
        Some("agent"),
    )?;
    Ok(lab::Outcome {
        group,
        means: vec![mean],
        results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calc;

    #[test]
    fn task_table_is_consistent() {
        for t in TASKS {
            assert_eq!(calc::eval(t.expr).unwrap(), t.answer, "{}", t.question);
        }
    }

    #[test]
    fn parses_actions() {
        assert_eq!(parse_action("CALC: 2*3"), Action::Calc("2*3".into()));
        assert_eq!(
            parse_action("  answer: 1,200 dollars"),
            Action::Answer(1200)
        );
        assert_eq!(parse_action("\nANSWER: -4"), Action::Answer(-4));
        assert!(matches!(parse_action("I think it's 5"), Action::Invalid(_)));
    }

    #[test]
    fn memory_window_drops_lines() {
        let h: Vec<String> = (0..5).map(|i| format!("CALC: {i} -> {i}")).collect();
        let p = actor_prompt(&TASKS[0], &h, Some(2));
        assert!(p.contains("3 earlier lines dropped"));
        assert!(!p.contains("CALC: 0 -> 0"));
        assert!(p.contains("CALC: 4 -> 4"));
        assert!(lookup_task(&p).is_some());
    }
}
