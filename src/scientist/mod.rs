//! Machine Scientist: an explicit model of the performance landscape, fitted
//! from the notebook, and an experiment selector that chooses the next
//! single-variable trial by expected information gain × practical value ÷
//! cost.
//!
//! Layers (each inspectable on its own):
//! - `fit`       robust lines, bootstrap intervals, Wilson intervals
//! - `analyze`   notebook → `BeliefState` + `Surrogate` (pure)
//! - `beliefs`   the serializable belief state (`machine beliefs`)
//! - `surrogate` answers "what will configuration X cost?"
//! - `selector`  ranks candidate experiments; a `Planner` for the shared loop
//! - `report`    markdown research note (`machine report`)
//!
//! It never searches over several variables at once: every candidate is an
//! `autopilot::ExperimentSpec`, validated single-variable, run through the
//! same execution path and scored by the same scorer.

pub mod analyze;
pub mod beliefs;
pub mod fit;
pub mod report;
pub mod selector;
pub mod surrogate;

/// Config key of the baseline agent (greedy, no critic, full history, no cache).
pub const BASELINE_KEY: &str = "critic=none window=all cache=false temp=0";

#[cfg(test)]
mod tests;
