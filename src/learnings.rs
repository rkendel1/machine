//! Structured learning records: one per autopilot trial, in
//! `learnings.jsonl` next to the notebook. The notebook keeps the raw rows;
//! a learning is the interpretation — what was predicted, what happened,
//! what it means, and what the numbers suggest next.

use crate::metrics;
use crate::notebook::Notebook;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Learning {
    /// Always "learning".
    #[serde(rename = "type")]
    pub kind: String,
    /// ISO-8601, UTC.
    pub ts: String,
    /// The equivalent command line, predictions included.
    pub experiment: String,
    /// metric → number (one point) or list (one per sweep point).
    pub prediction: BTreeMap<String, Value>,
    pub actual: BTreeMap<String, Value>,
    /// metric → worst error over its points, in percent (rates: percentage points).
    pub error_pct: BTreeMap<String, f64>,
    /// Every prediction within tolerance.
    pub passed: bool,
    /// 1–3 sentences, plain English, numbers included.
    pub learning: String,
    pub next_priority: String,
    // Provenance (beyond the minimal format): links back to the notebook.
    /// Notebook group id of the trial's rows.
    pub group: String,
    /// Proposer key, e.g. "sweep:cache_prompt".
    pub key: String,
    /// Result of the compare-style multi-setting check against the baseline.
    pub single_variable: bool,
    pub backend: String,
    pub simulated: bool,
}

pub fn path(nb: &Notebook) -> PathBuf {
    nb.dir().join("learnings.jsonl")
}

pub fn append(nb: &Notebook, l: &Learning) -> Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path(nb))?;
    writeln!(f, "{}", serde_json::to_string(l)?)?;
    Ok(())
}

pub fn load(nb: &Notebook) -> Result<Vec<Learning>> {
    let Ok(f) = fs::File::open(path(nb)) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(l) => out.push(l),
            Err(e) => eprintln!("warning: skipping learnings line {}: {e}", i + 1),
        }
    }
    Ok(out)
}

/// One number, or a list when there are several points.
pub fn value_or_list(xs: &[f64]) -> Value {
    let round = |x: f64| (x * 1000.0).round() / 1000.0;
    match xs {
        [x] => Value::from(round(*x)),
        xs => Value::from(xs.iter().map(|x| round(*x)).collect::<Vec<_>>()),
    }
}

/// Unix seconds → "YYYY-MM-DDTHH:MM:SSZ" (proleptic Gregorian, UTC).
pub fn iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn fmt_value(metric: &str, v: &Value) -> String {
    let one = |x: f64| metrics::fmt(metric, x);
    match v {
        Value::Number(n) => one(n.as_f64().unwrap_or(0.0)),
        Value::Array(xs) => xs
            .iter()
            .map(|x| one(x.as_f64().unwrap_or(0.0)))
            .collect::<Vec<_>>()
            .join(", "),
        _ => v.to_string(),
    }
}

pub fn print(ls: &[Learning], tail: usize) {
    if ls.is_empty() {
        println!("No learnings yet. `machine autopilot --dry-run` shows what it would try.");
        return;
    }
    let passed = ls.iter().filter(|l| l.passed).count();
    println!(
        "═══ LEARNINGS ═══  {} trial(s), {passed} with every prediction within tolerance\n",
        ls.len()
    );
    for l in ls.iter().skip(ls.len().saturating_sub(tail)) {
        println!(
            "{}  {}  {}{}{}",
            l.ts,
            if l.passed { "✓" } else { "✗" },
            l.key,
            if l.simulated { "  [SIMULATED]" } else { "" },
            if l.single_variable {
                ""
            } else {
                "  [CONFOUNDED]"
            }
        );
        println!("  $ {}", l.experiment);
        for (m, p) in &l.prediction {
            let a = l
                .actual
                .get(m)
                .map(|a| fmt_value(m, a))
                .unwrap_or("—".into());
            println!(
                "    {m:<16} predicted {:<24} measured {:<24} worst error {:.1}%",
                fmt_value(m, p),
                a,
                l.error_pct.get(m).copied().unwrap_or(0.0)
            );
        }
        println!("  learning: {}", l.learning);
        println!("  next    : {}\n", l.next_priority);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso8601(1_792_713_600), "2026-10-23T00:00:00Z");
    }

    #[test]
    fn values() {
        assert_eq!(value_or_list(&[1.23456]), Value::from(1.235));
        assert_eq!(value_or_list(&[1.0, 2.0]).as_array().unwrap().len(), 2);
    }
}
