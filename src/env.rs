//! `machine env`: what this machine actually is. Reports only what can be
//! read; anything else is left for the student to paste in.

use crate::memory;
use std::process::Command;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

pub fn print(llama_url: &str, ollama_url: &str) {
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    println!("═══ ENVIRONMENT ═══");
    println!(
        "os        : {}",
        System::long_os_version().unwrap_or_else(|| "?".into())
    );
    let brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_default();
    println!(
        "cpu       : {}  ({} physical / {} logical cores)",
        if brand.is_empty() { "?".into() } else { brand },
        System::physical_core_count()
            .map(|n| n.to_string())
            .unwrap_or("?".into()),
        sys.cpus().len()
    );
    println!("ram       : {}", memory::fmt_bytes(sys.total_memory()));
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(o) if o.status.success() => {
            for l in String::from_utf8_lossy(&o.stdout).lines() {
                println!("gpu       : {l}");
            }
        }
        _ => println!(
            "gpu       : no nvidia-smi found (Apple Silicon: Metal is used by llama.cpp/Ollama automatically; AMD: paste `rocm-smi`)"
        ),
    }

    let agent = crate::backend::http_agent();
    let probe = |url: String| -> String {
        match agent
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
        {
            Ok(r) => format!(
                "up ({})",
                r.into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(80)
                    .collect::<String>()
                    .trim()
            ),
            Err(ureq::Error::Status(c, _)) => format!("responding with HTTP {c}"),
            Err(_) => "not reachable".into(),
        }
    };
    println!(
        "llama-srv : {}  {}",
        llama_url,
        probe(format!("{}/health", llama_url.trim_end_matches('/')))
    );
    println!(
        "ollama    : {}  {}",
        ollama_url,
        probe(format!("{}/api/version", ollama_url.trim_end_matches('/')))
    );

    for name in ["llama-server", "ollama"] {
        if let Some((pid, desc)) = memory::resolve(&memory::WatchTarget::Name(name.into())) {
            let rss = memory::rss_bytes(pid)
                .map(memory::fmt_bytes)
                .unwrap_or("?".into());
            let hwm = memory::vm_hwm_bytes(pid)
                .map(|b| format!(", lifetime peak {}", memory::fmt_bytes(b)))
                .unwrap_or_default();
            println!("process   : {desc} rss {rss}{hwm}");
        }
    }
    println!("\nPaste this block when asking for help: predictions depend on these numbers.");
}
