//! Small, transparent estimators. Every number the scientist reports comes
//! from one of these, so they are deliberately simple:
//!
//! - lines: Theil–Sen (median of pairwise slopes; intercept = median of
//!   y − slope·x). Robust to warm-up outliers without any tuning constant.
//! - uncertainty: percentile bootstrap, `B` resamples, fixed seed, so the
//!   same notebook always gives the same intervals.
//! - proportions: Wilson score interval (well-behaved at 0% and 100%).

use crate::backend::mock::Rng;
use serde::{Deserialize, Serialize};

/// Bootstrap resamples. 200 gives stable 95% percentile intervals for the
/// sample sizes a notebook has (tens to hundreds of rows) in milliseconds.
pub const B: usize = 200;
/// Fixed seed: identical notebook → identical beliefs.
pub const SEED: u64 = 7;

/// An estimate with a 95% interval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Est {
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Est {
    pub fn half_width(&self) -> f64 {
        (self.hi - self.lo) / 2.0
    }
    /// Half-width relative to the estimate (∞ when the estimate is ~0).
    pub fn rel_width(&self) -> f64 {
        if self.value.abs() < 1e-9 {
            f64::INFINITY
        } else {
            self.half_width() / self.value.abs()
        }
    }
    pub fn excludes_zero(&self) -> bool {
        self.lo > 0.0 || self.hi < 0.0
    }
}

/// A robust straight line y = intercept + slope·x with bootstrap intervals.
#[derive(Debug, Clone)]
pub struct Line {
    pub intercept: Est,
    pub slope: Est,
    pub n: usize,
    /// Distinct x values (a line through one x value is not a line).
    pub distinct_x: usize,
    /// Measured x range: predictions outside it are extrapolations.
    pub x_min: f64,
    pub x_max: f64,
    pub r2: f64,
    /// Median absolute residual: typical miss of the line, in y units.
    pub mad_resid: f64,
    /// Bootstrap (intercept, slope) draws, for propagating uncertainty.
    pub draws: Vec<(f64, f64)>,
}

impl Line {
    pub fn at(&self, x: f64) -> f64 {
        self.intercept.value + self.slope.value * x
    }
    /// 95% interval of the line's value at x, from the joint draws.
    pub fn at_ci(&self, x: f64) -> Est {
        let ys: Vec<f64> = self.draws.iter().map(|(a, b)| a + b * x).collect();
        let (lo, hi) = percentile_ci(&ys);
        Est {
            value: self.at(x),
            lo,
            hi,
        }
    }
}

pub fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    })
}

/// Theil–Sen point estimate. None if fewer than two distinct x values.
pub fn theil_sen(pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut slopes = Vec::with_capacity(pts.len() * pts.len() / 2);
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let dx = pts[j].0 - pts[i].0;
            if dx.abs() > 1e-9 {
                slopes.push((pts[j].1 - pts[i].1) / dx);
            }
        }
    }
    let b = median(&mut slopes)?;
    let mut resid: Vec<f64> = pts.iter().map(|(x, y)| y - b * x).collect();
    let a = median(&mut resid)?;
    Some((a, b))
}

fn percentile_ci(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| crate::stats::percentile(&s, p);
    (q(0.025), q(0.975))
}

fn resample<T: Copy>(xs: &[T], rng: &mut Rng) -> Vec<T> {
    (0..xs.len())
        .map(|_| xs[(rng.next_u64() % xs.len() as u64) as usize])
        .collect()
}

/// Fit a robust line with bootstrap intervals. Needs ≥3 points over ≥2 distinct x.
pub fn fit_line(pts: &[(f64, f64)]) -> Option<Line> {
    let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if pts.len() < 3 || xs.len() < 2 {
        return None;
    }
    // Theil–Sen is O(n²); thin very large inputs deterministically.
    let pts: Vec<(f64, f64)> = if pts.len() > 400 {
        let step = pts.len().div_ceil(400);
        pts.iter().step_by(step).copied().collect()
    } else {
        pts.to_vec()
    };
    let (a, b) = theil_sen(&pts)?;
    let mut rng = Rng::new(SEED);
    let draws: Vec<(f64, f64)> = (0..B)
        .filter_map(|_| theil_sen(&resample(&pts, &mut rng)))
        .collect();
    let (alo, ahi) = percentile_ci(&draws.iter().map(|d| d.0).collect::<Vec<_>>());
    let (blo, bhi) = percentile_ci(&draws.iter().map(|d| d.1).collect::<Vec<_>>());
    let my = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
    let ss_tot: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
    let ss_res: f64 = pts.iter().map(|p| (p.1 - (a + b * p.0)).powi(2)).sum();
    let mut abs_res: Vec<f64> = pts.iter().map(|p| (p.1 - (a + b * p.0)).abs()).collect();
    Some(Line {
        intercept: Est {
            value: a,
            lo: alo,
            hi: ahi,
        },
        slope: Est {
            value: b,
            lo: blo,
            hi: bhi,
        },
        n: pts.len(),
        distinct_x: xs.len(),
        x_min: xs[0],
        x_max: xs[xs.len() - 1],
        r2: if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        },
        mad_resid: median(&mut abs_res).unwrap_or(0.0),
        draws,
    })
}

/// Mean with a bootstrap 95% interval. None for an empty sample.
pub fn mean_ci(xs: &[f64]) -> Option<Est> {
    if xs.is_empty() {
        return None;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    if xs.len() == 1 {
        return Some(Est {
            value: mean,
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        });
    }
    let mut rng = Rng::new(SEED);
    let ms: Vec<f64> = (0..B)
        .map(|_| {
            let r = resample(xs, &mut rng);
            r.iter().sum::<f64>() / r.len() as f64
        })
        .collect();
    let (lo, hi) = percentile_ci(&ms);
    Some(Est {
        value: mean,
        lo,
        hi,
    })
}

/// Difference of means (b − a) with a two-sample bootstrap 95% interval.
pub fn diff_ci(a: &[f64], b: &[f64]) -> Option<Est> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let m = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let mut rng = Rng::new(SEED);
    let ds: Vec<f64> = (0..B)
        .map(|_| m(&resample(b, &mut rng)) - m(&resample(a, &mut rng)))
        .collect();
    let (lo, hi) = percentile_ci(&ds);
    Some(Est {
        value: m(b) - m(a),
        lo,
        hi,
    })
}

/// Wilson score 95% interval for k successes in n trials.
pub fn wilson(k: f64, n: f64) -> Est {
    if n <= 0.0 {
        return Est {
            value: f64::NAN,
            lo: 0.0,
            hi: 1.0,
        };
    }
    let z = 1.96_f64;
    let p = k / n;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / denom;
    Est {
        value: p,
        lo: (centre - half).max(0.0),
        hi: (centre + half).min(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_line_with_outlier() {
        let mut pts: Vec<(f64, f64)> = (0..30)
            .map(|i| (i as f64 * 10.0, 25.0 + 6.0 * i as f64 * 10.0))
            .collect();
        pts.push((50.0, 5000.0)); // a warm-up outlier
        let l = fit_line(&pts).unwrap();
        assert!((l.slope.value - 6.0).abs() < 1e-6, "{}", l.slope.value);
        assert!((l.intercept.value - 25.0).abs() < 1e-6);
        assert!(l.slope.lo <= 6.0 && l.slope.hi >= 6.0);
    }

    #[test]
    fn needs_two_x_values() {
        assert!(fit_line(&[(1.0, 2.0), (1.0, 3.0), (1.0, 4.0)]).is_none());
    }

    #[test]
    fn wilson_edges() {
        let w = wilson(0.0, 10.0);
        assert_eq!(w.lo, 0.0);
        assert!(w.hi > 0.2 && w.hi < 0.35);
        let w = wilson(5.0, 10.0);
        assert!((w.value - 0.5).abs() < 1e-9 && w.lo > 0.2 && w.hi < 0.8);
    }

    #[test]
    fn deterministic_bootstrap() {
        let xs = [1.0, 2.0, 3.0, 4.0, 10.0];
        assert_eq!(mean_ci(&xs), mean_ci(&xs));
        let d = diff_ci(&[1.0, 1.1, 0.9, 1.0], &[2.0, 2.1, 1.9, 2.0]).unwrap();
        assert!(d.excludes_zero());
    }
}
