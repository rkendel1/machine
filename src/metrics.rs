//! Named metrics. Every number the tool prints has a name here, so predictions,
//! comparisons and sweeps all speak the same vocabulary.

use crate::backend::Generation;
use std::collections::BTreeMap;

pub const MIB: f64 = 1024.0 * 1024.0;

/// (name, unit, what it measures)
pub const RUN_METRICS: &[(&str, &str, &str)] = &[
    (
        "decode_tok_s",
        "tok/s",
        "completion tokens / decode time (server-reported)",
    ),
    ("prefill_tok_s", "tok/s", "prompt tokens / prefill time"),
    (
        "prefill_ms",
        "ms",
        "server time spent processing the prompt",
    ),
    ("decode_ms", "ms", "server time spent generating tokens"),
    ("wall_ms", "ms", "client-measured end-to-end request time"),
    (
        "overhead_ms",
        "ms",
        "wall - prefill - decode (HTTP, JSON, scheduling, load)",
    ),
    (
        "load_ms",
        "ms",
        "model load time reported by the server (Ollama)",
    ),
    (
        "prompt_tokens",
        "tok",
        "prompt tokens evaluated (cached prefix excluded)",
    ),
    ("completion_tokens", "tok", "tokens generated"),
    (
        "start_rss_mib",
        "MiB",
        "watched process RSS before the call",
    ),
    ("peak_rss_mib", "MiB", "sampled peak RSS during the call"),
    ("rss_delta_mib", "MiB", "peak - start"),
];

pub const AGENT_METRICS: &[(&str, &str, &str)] = &[
    ("success", "0/1", "final answer matched ground truth"),
    ("steps", "n", "agent loop iterations"),
    ("calls", "n", "inference calls (actor + critic)"),
    ("actor_calls", "n", "inference calls that chose actions"),
    ("critic_calls", "n", "inference calls spent verifying"),
    (
        "format_errors",
        "n",
        "replies that matched neither CALC: nor ANSWER:",
    ),
    (
        "total_tokens",
        "tok",
        "prompt + completion tokens over all calls",
    ),
    ("prompt_tokens", "tok", "prompt tokens over all calls"),
    (
        "completion_tokens",
        "tok",
        "completion tokens over all calls",
    ),
    ("wall_ms", "ms", "end-to-end episode time"),
    ("infer_ms", "ms", "sum of per-call wall time"),
    ("critic_ms", "ms", "wall time spent in critic calls"),
    (
        "critic_false_reject",
        "n",
        "critic said NO to a correct answer",
    ),
    (
        "critic_false_accept",
        "n",
        "critic said YES to a wrong answer",
    ),
    ("peak_rss_mib", "MiB", "sampled peak RSS over the episode"),
];

pub fn unit(name: &str) -> &'static str {
    RUN_METRICS
        .iter()
        .chain(AGENT_METRICS)
        .find(|(n, _, _)| *n == name)
        .map(|(_, u, _)| *u)
        .unwrap_or("")
}

pub fn is_known(name: &str) -> bool {
    RUN_METRICS
        .iter()
        .chain(AGENT_METRICS)
        .any(|(n, _, _)| *n == name)
}

/// Higher is better for these; lower is better for everything else.
pub fn higher_is_better(name: &str) -> bool {
    matches!(name, "decode_tok_s" | "prefill_tok_s" | "success")
}

pub fn from_generation(g: &Generation) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Option<f64>| {
        if let Some(v) = v {
            m.insert(k.to_string(), v);
        }
    };
    put("decode_tok_s", Some(g.decode_tok_s()));
    put("prefill_tok_s", g.prefill_tok_s());
    put("prefill_ms", g.prefill_ms);
    put("decode_ms", g.decode_ms);
    put("wall_ms", Some(g.wall_ms));
    put("overhead_ms", g.overhead_ms());
    put("load_ms", g.load_ms);
    put("prompt_tokens", Some(g.prompt_tokens as f64));
    put("completion_tokens", Some(g.completion_tokens as f64));
    put("start_rss_mib", g.start_rss_bytes.map(|b| b as f64 / MIB));
    put("peak_rss_mib", g.peak_rss_bytes.map(|b| b as f64 / MIB));
    if let (Some(s), Some(p)) = (g.start_rss_bytes, g.peak_rss_bytes) {
        put("rss_delta_mib", Some((p as f64 - s as f64) / MIB));
    }
    m
}

pub fn fmt(name: &str, v: f64) -> String {
    match unit(name) {
        "0/1" => format!("{:.0}%", v * 100.0),
        "tok" | "n" if v.fract() == 0.0 => format!("{v:.0}"),
        "tok" | "n" => format!("{v:.2}"),
        "ms" if v >= 10_000.0 => format!("{:.2} s", v / 1000.0),
        "ms" => format!("{v:.1} ms"),
        "tok/s" => format!("{v:.1} tok/s"),
        "MiB" => format!("{v:.1} MiB"),
        _ => format!("{v:.3}"),
    }
}

/// Short form for tables: number only, more digits when small.
pub fn short(v: f64) -> String {
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}
