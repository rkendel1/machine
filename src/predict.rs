//! Predict before you run. Predictions are written down *before* the
//! experiment executes and scored against the measured mean afterwards.

use crate::metrics;
use crate::notebook::PredictionResult;
use anyhow::{Result, bail};
use std::io::{self, BufRead, IsTerminal, Write};

pub const DEFAULT_TOLERANCE_PCT: f64 = 20.0;

/// A prediction for one metric; one value per sweep point (or one for a plain run).
#[derive(Debug, Clone)]
pub struct Prediction {
    pub metric: String,
    pub values: Vec<f64>,
}

/// Parse `metric=v` or `metric=v1,v2,v3`.
pub fn parse(s: &str) -> Result<Prediction> {
    let Some((m, v)) = s.split_once('=') else {
        bail!("prediction must look like metric=value, got {s:?}");
    };
    let m = m.trim();
    if !metrics::is_known(m) {
        bail!("unknown metric {m:?}; see `machine metrics`");
    }
    let values = v
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("bad number in {s:?}: {e}"))?;
    Ok(Prediction {
        metric: m.to_string(),
        values,
    })
}

pub fn interactive() -> bool {
    io::stdin().is_terminal()
}

pub fn ask_line(q: &str) -> Option<String> {
    print!("{q}");
    io::stdout().flush().ok()?;
    let mut s = String::new();
    io::stdin().lock().read_line(&mut s).ok()?;
    Some(s.trim().to_string())
}

/// Ask the student for predictions at each point. Blank = skip that metric
/// (no credit). Returns None for everything skipped.
pub fn ask(metrics_to_ask: &[&str], points: &[String]) -> Vec<Prediction> {
    println!("\n── PREDICT FIRST ─────────────────────────────────────────────");
    println!("Write down what you expect before anything runs. Blank skips (no credit).");
    let mut out = Vec::new();
    for m in metrics_to_ask {
        let mut values = Vec::new();
        'points: for pt in points {
            let label = if pt.is_empty() {
                String::new()
            } else {
                format!(" at {pt}")
            };
            loop {
                let ans = ask_line(&format!("  predicted {m}{label} [{}]: ", metrics::unit(m)))
                    .unwrap_or_default();
                if ans.is_empty() {
                    break 'points; // skipping one point skips the metric
                }
                match ans.parse::<f64>() {
                    Ok(v) => {
                        values.push(v);
                        break;
                    }
                    Err(_) => println!("    a number please"),
                }
            }
        }
        if values.len() == points.len() {
            out.push(Prediction {
                metric: m.to_string(),
                values,
            });
        }
    }
    let why = ask_line("  one line: WHY do you expect that? ").unwrap_or_default();
    if !why.is_empty() {
        println!("  noted. We'll check the reasoning against the numbers.");
    }
    out
}

pub fn score(metric: &str, predicted: f64, actual: f64, tol_pct: f64) -> PredictionResult {
    // Rates (0..1) are scored in percentage points: predicting 40% when the
    // truth is 30% is a 10-point miss, not a 33% miss — and 0% stays scoreable.
    let error_pct = if metrics::unit(metric.split('@').next().unwrap_or(metric)) == "0/1" {
        (predicted - actual).abs() * 100.0
    } else if actual == 0.0 {
        if predicted == 0.0 { 0.0 } else { 100.0 }
    } else {
        (predicted - actual).abs() / actual.abs() * 100.0
    };
    PredictionResult {
        metric: metric.to_string(),
        predicted,
        actual,
        error_pct,
        passed: error_pct <= tol_pct,
    }
}

pub fn print_results(rs: &[PredictionResult], context: &str) {
    if rs.is_empty() {
        return;
    }
    println!("\n── PREDICTION vs MEASURED {context}");
    for r in rs {
        let base = r.metric.split('@').next().unwrap_or(&r.metric);
        println!(
            "  {:<24} predicted {:>12}  measured {:>12}  error {:>5.1}%  {}",
            r.metric,
            metrics::fmt(base, r.predicted),
            metrics::fmt(base, r.actual),
            r.error_pct,
            if r.passed {
                "✓ within tolerance"
            } else {
                "✗ off — find the log line that explains the gap"
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_forms() {
        let p = parse("decode_tok_s=30").unwrap();
        assert_eq!(p.values, vec![30.0]);
        let p = parse("wall_ms = 100, 200,300").unwrap();
        assert_eq!(p.values.len(), 3);
        assert!(parse("nonsense=1").is_err());
        assert!(parse("decode_tok_s").is_err());
    }

    #[test]
    fn scoring() {
        assert!(score("wall_ms", 110.0, 100.0, 20.0).passed);
        assert!(!score("wall_ms", 150.0, 100.0, 20.0).passed);
        let r = score("success", 0.4, 0.3, 20.0);
        assert!((r.error_pct - 10.0).abs() < 1e-9 && r.passed);
    }
}
