//! The lab notebook: an append-only JSONL file of every run and prediction.
//!
//! Append-only means nothing is overwritten, so a baseline is always
//! recoverable and every comparison can be re-done from raw rows.
//! Location: $MACHINE_HOME, else ./.machine/

use crate::backend::GenParams;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// One inference call.
    Run,
    /// One agent episode (many inference calls summarised).
    Agent,
    /// A prediction scored against the mean of a group of runs.
    Prediction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub metric: String,
    pub predicted: f64,
    pub actual: f64,
    pub error_pct: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub ts: u64,
    pub kind: Kind,
    /// Groups repeated runs / sweep points / an agent batch together.
    pub group: String,
    pub lesson: Option<u8>,
    pub experiment: String,
    pub label: Option<String>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    pub backend: String,
    pub model: Option<String>,
    pub params: Option<GenParams>,
    /// For sweeps: the variable and value this row was run at.
    pub var: Option<String>,
    pub value: Option<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    pub simulated: bool,
    #[serde(default)]
    pub predictions: Vec<PredictionResult>,
    #[serde(default)]
    pub text_preview: String,
    /// Warm-up rows are kept for transparency but excluded from summaries.
    #[serde(default)]
    pub warmup: bool,
}

pub fn home() -> PathBuf {
    std::env::var_os("MACHINE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".machine"))
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn new_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{:x}", nanos)
}

pub struct Notebook {
    path: PathBuf,
}

impl Notebook {
    pub fn open() -> Result<Self> {
        Self::open_at(home())
    }

    pub fn open_at(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        Ok(Self {
            path: dir.join("notebook.jsonl"),
        })
    }

    /// Directory holding the notebook (sibling files such as learnings.jsonl live here).
    pub fn dir(&self) -> PathBuf {
        self.path.parent().map(PathBuf::from).unwrap_or_else(home)
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn append(&self, r: &Record) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(r)?)?;
        Ok(())
    }

    pub fn all(&self) -> Result<Vec<Record>> {
        let Ok(f) = fs::File::open(&self.path) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Record>(&line) {
                Ok(r) => out.push(r),
                Err(e) => eprintln!("warning: skipping notebook line {}: {e}", i + 1),
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Profile {
    /// 0 = none, 1 = some, 2 = comfortable
    pub rust: u8,
    pub ownership: u8,
    pub async_rust: u8,
    pub cli: u8,
    pub local_inference: u8,
    pub tools_used: Vec<String>,
    pub hardware_notes: String,
}

impl Profile {
    pub fn path() -> PathBuf {
        home().join("profile.json")
    }
    pub fn load() -> Option<Self> {
        let s = fs::read_to_string(Self::path()).ok()?;
        serde_json::from_str(&s).ok()
    }
    pub fn save(&self) -> Result<()> {
        fs::create_dir_all(home())?;
        fs::write(Self::path(), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
    pub fn beginner_rust(&self) -> bool {
        self.rust == 0 || self.ownership == 0
    }
}
