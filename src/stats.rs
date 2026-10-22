//! Summary statistics for repeated runs. Kept deliberately small and explicit.

#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(xs: &[f64]) -> Option<Self> {
        if xs.is_empty() {
            return None;
        }
        let n = xs.len();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        Some(Self {
            n,
            mean,
            sd: var.sqrt(),
            min: s[0],
            p50: percentile(&s, 0.50),
            p95: percentile(&s, 0.95),
            max: s[n - 1],
        })
    }

    /// Coefficient of variation (sd / mean). > ~5% means noise can swamp small effects.
    pub fn cv(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            self.sd / self.mean.abs()
        }
    }

    pub fn stderr(&self) -> f64 {
        self.sd / (self.n as f64).sqrt()
    }
}

/// Linear-interpolated percentile on sorted data.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Is the difference between two groups bigger than their noise?
/// Uses |Δmean| / sqrt(se_a² + se_b²) (Welch's t statistic). We report it and
/// call |t| >= 2 "distinguishable" — a rule of thumb, not a p-value.
pub fn welch_t(a: &Summary, b: &Summary) -> Option<f64> {
    if a.n < 2 || b.n < 2 {
        return None;
    }
    let se = (a.stderr().powi(2) + b.stderr().powi(2)).sqrt();
    if se == 0.0 {
        return Some(if a.mean == b.mean { 0.0 } else { f64::INFINITY });
    }
    Some((b.mean - a.mean) / se)
}

pub fn verdict(t: Option<f64>) -> &'static str {
    match t {
        None => "need ≥2 runs each",
        Some(t) if t.abs() >= 2.0 => "real difference (|t|≥2)",
        Some(_) => "within noise",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_basics() {
        let s = Summary::of(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.p50, 3.0);
        assert!((s.sd - 1.5811).abs() < 1e-3);
        assert!((s.p95 - 4.8).abs() < 1e-9);
    }

    #[test]
    fn welch_separates() {
        let a = Summary::of(&[10.0, 10.1, 9.9, 10.0]).unwrap();
        let b = Summary::of(&[12.0, 12.1, 11.9, 12.0]).unwrap();
        assert!(welch_t(&a, &b).unwrap() > 2.0);
        let c = Summary::of(&[10.0, 12.0, 8.0, 11.0]).unwrap();
        let d = Summary::of(&[10.5, 11.0, 9.0, 10.0]).unwrap();
        assert!(welch_t(&c, &d).unwrap().abs() < 2.0);
    }
}
