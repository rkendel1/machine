//! Resident-memory sampling for the process that actually holds the model.
//!
//! With an HTTP backend the weights and KV cache live in the *server* process,
//! not in `machine`. So we sample that process's RSS on a background thread
//! while a request is in flight. This is a *sampled* peak: spikes shorter than
//! the interval can be missed. On Linux, `VmHWM` gives the true lifetime peak.
//!
//! Note: llama.cpp mmaps model files by default, so RSS counts only pages that
//! have been touched. The first request after start can show RSS growth that
//! later requests do not.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::JoinHandle;
use std::time::Duration;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Debug, Clone)]
pub enum WatchTarget {
    /// Don't sample memory.
    None,
    /// This process (in-process backends, e.g. the mock).
    SelfProcess,
    Pid(u32),
    /// Largest-RSS process whose name contains this string.
    Name(String),
}

pub fn resolve(target: &WatchTarget) -> Option<(u32, String)> {
    match target {
        WatchTarget::None => None,
        WatchTarget::SelfProcess => {
            let pid = std::process::id();
            Some((pid, "self".into()))
        }
        WatchTarget::Pid(p) => Some((*p, format!("pid {p}"))),
        WatchTarget::Name(name) => {
            let mut sys = System::new();
            sys.refresh_processes_specifics(
                ProcessesToUpdate::All,
                true,
                ProcessRefreshKind::nothing().with_memory(),
            );
            sys.processes_by_name(name.as_ref())
                .max_by_key(|p| p.memory())
                .map(|p| {
                    (
                        p.pid().as_u32(),
                        format!("{} (pid {})", p.name().to_string_lossy(), p.pid()),
                    )
                })
        }
    }
}

pub fn rss_bytes(pid: u32) -> Option<u64> {
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::nothing().with_memory(),
    );
    sys.process(pid).map(|p| p.memory())
}

/// Linux only: lifetime peak RSS (VmHWM) in bytes.
pub fn vm_hwm_bytes(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

pub struct Sampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    start: Option<u64>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    pub fn start(pid: Option<u32>, interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let Some(pid) = pid else {
            return Self {
                stop,
                peak,
                start: None,
                handle: None,
            };
        };
        let start = rss_bytes(pid);
        peak.store(start.unwrap_or(0), Ordering::Relaxed);
        let (s, pk) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            let mut sys = System::new();
            let pid = Pid::from_u32(pid);
            while !s.load(Ordering::Relaxed) {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    ProcessRefreshKind::nothing().with_memory(),
                );
                if let Some(p) = sys.process(pid) {
                    pk.fetch_max(p.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(interval);
            }
        });
        Self {
            stop,
            peak,
            start,
            handle: Some(handle),
        }
    }

    /// Returns (rss at start, sampled peak during the call).
    pub fn finish(mut self) -> (Option<u64>, Option<u64>) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
            (self.start, Some(self.peak.load(Ordering::Relaxed)))
        } else {
            (None, None)
        }
    }
}

pub fn fmt_bytes(b: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let mib = b as f64 / MIB;
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{mib:.1} MiB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_self() {
        let (pid, _) = resolve(&WatchTarget::SelfProcess).unwrap();
        let s = Sampler::start(Some(pid), Duration::from_millis(5));
        std::thread::sleep(Duration::from_millis(30));
        let (start, peak) = s.finish();
        assert!(start.unwrap() > 0);
        assert!(peak.unwrap() >= start.unwrap());
    }
}
