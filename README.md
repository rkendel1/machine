# machine

A Rust lab for learning to **design, measure and optimize agentic loops on local
inference** — by predicting what will happen, running it, and reading the numbers.

```
predict → run small experiment → compare metrics → explain → next single change
```

Every command asks for your prediction first, runs against a real local model
(llama.cpp `llama-server` or Ollama), splits each call into **prefill / decode /
overhead / memory**, appends raw rows to an append-only notebook, and scores the
prediction. A lesson only counts as passed once you can predict its numbers.

**See a real session:** [`examples/`](examples/README.md) — Qwen2.5-0.5B on a
2-vCPU box, all six lessons, with full transcripts and the raw notebook. CI
re-runs that session on a GitHub runner against a cached real model on every push.

## Install

```sh
git clone https://github.com/rkendel1/machine && cd machine
cargo build --release          # Rust 1.88+
alias machine=./target/release/machine
```

Get a model server (any one):

```sh
# llama.cpp (recommended: reports prefill/decode timings precisely)
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 2048 --port 8080
export MACHINE_BACKEND=llama-server MACHINE_TEMPLATE=chatml

# or Ollama
ollama pull qwen2.5:0.5b
export MACHINE_BACKEND=ollama MACHINE_MODEL=qwen2.5:0.5b
```

No model yet? The default `--backend mock` is a deterministic simulator for
learning the harness. Its output is labelled `SIMULATED` everywhere and it is
never evidence about your hardware.

## Use

```sh
machine start          # what you already know → adapts lesson depth
machine env            # CPU, RAM, GPU, which backends are up — paste this when asking for help
machine lesson         # your current lesson card
```

| Lesson | Command | What you learn to predict |
|---|---|---|
| 1. Minimal inference loop | `machine run --max-tokens 64` | prefill vs decode time, tok/s, RSS |
| 2. Measurement discipline | `machine run --repeat 10 --warmup 1` | variance (cv%), warm-up effects, the smallest detectable change |
| 3. Single-variable experiments | `machine sweep --var max-tokens --values 16,32,64` · `compare --label q4 --label q8` | slopes: ms/token, prefill vs prompt length, quantization trade-offs |
| 4. Agentic wrapper | `machine agent --verbose` | calls × per-call cost, prompt growth, format failures |
| 5. Critic / verifier | `machine agent --critic model` (vs `none`, `oracle`) | whether verification buys more accuracy than it costs |
| 6. Closing the loop | `machine next` · `machine wrap-up` | choosing the next experiment from recorded numbers |

```sh
machine progress       # lessons passed by prediction accuracy
machine next           # one concrete untried experiment, with the number that motivates it
machine wrap-up        # end-of-session check, weakest idea, WHAT SHOULD I BUILD NEXT?
machine metrics        # every metric name you can --predict or --metric
machine notebook       # where the raw data lives, last rows
```

### Predictions

Interactive terminals are asked before anything runs. Or pass them:

```sh
machine run --predict decode_tok_s=28 --predict wall_ms=2500
machine sweep --var max-tokens --values 16,32,64 --predict wall_ms=780,1340,2490
machine agent --predict success=0.5 --predict calls=3
```

Tolerance is ±20% (`--tolerance`); success rates are scored in percentage points.

### Load-time knobs

Quantization, `-t` threads, `-ngl` GPU layers and `-c` context are fixed when the
server starts. Change one, restart, and tag:

```sh
machine run --repeat 8 --warmup 1 --ignore-eos --label q4 --tag quant=q4_k_m
# restart llama-server with the q8_0 file
machine run --repeat 8 --warmup 1 --ignore-eos --label q8 --tag quant=q8_0
machine compare --label q4 --label q8
```

`compare` warns when two labels differ in more than one setting.

## What it measures

| metric | source |
|---|---|
| `prefill_ms`, `decode_ms` | server timings (`timings.prompt_ms` / `predicted_ms`; Ollama `prompt_eval_duration` / `eval_duration`) |
| `decode_tok_s`, `prefill_tok_s` | tokens ÷ the matching phase time |
| `wall_ms`, `overhead_ms` | client clock; overhead = wall − prefill − decode |
| `peak_rss_mib`, `rss_delta_mib` | sampled every 20 ms from the **server** process (that's where the weights and KV cache live) |
| `success`, `calls`, `critic_ms`, `format_errors`, `critic_false_accept` … | per agent episode, against ground-truth answers |

Things the tool points out when it sees them: early end-of-sequence
(use `--ignore-eos`), run-1 warm-up outliers, prompt-cache reuse silently shrinking
prefill, high cv%, and overhead that isn't prefill or decode.

## Layout

```
src/lab.rs            every inference call goes through Lab::infer (memory sampler + backend)
src/backend/          llama_server.rs, ollama.rs, mock.rs — one Generation record for all
src/experiment.rs     run / repeat / sweep (interleaved) / compare
src/agent.rs          observation → inference → tool → observation; critic modes; memory window
src/calc.rs           the agent's tool: exact integer arithmetic
src/stats.rs          mean, sd, p50/p95, cv, Welch t
src/predict.rs        prediction capture and scoring
src/advisor.rs        progress gating, `next`, `wrap-up`
src/lessons.rs        the six lesson cards
.machine/notebook.jsonl   append-only raw data (override with MACHINE_HOME)
```

## CI

`.github/workflows/ci.yml` runs fmt, clippy, tests and a simulator smoke test, then a
**real-model job**: it builds a pinned `llama-server` (static, portable CPU flags),
downloads Qwen2.5-0.5B-Instruct Q4_K_M and Q8_0 from Hugging Face, caches both
with `actions/cache`, runs [`examples/real-session.sh`](examples/real-session.sh),
writes the key tables to the job summary and uploads transcripts + notebook as an
artifact. Prediction misses on different hardware don't fail the build — they're data.
