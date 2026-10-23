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
machine autopilot      # run the cycle unattended (see Autopilot below)
machine scientist      # rank experiments by expected learning × value ÷ cost (see Scientist mode)
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

## Autopilot

`machine autopilot` runs the same cycle unattended: observe the notebook →
propose **one** single-variable experiment with numeric predictions → run it
through the normal `run` / `sweep` / `agent` code → score with the same scorer →
write a learning → repeat, until the budget or the untried levers run out.

```sh
machine autopilot --dry-run --budget 10   # show the plan, run nothing
machine autopilot --budget 30             # run it (Ctrl-C finishes the current trial, then stops)
machine autopilot --require-approval      # y / skip / quit before every trial
machine autopilot --focus cache           # prefill | decode | agent | quant | critic | cache
machine learnings                         # what it learned (--json for raw records)
```

What it will and won't do:

- **Deterministic proposer** (no LLM). It works through the advisor's untried
  levers in `machine next` order, then re-runs any trial whose prediction missed
  ("does it reproduce?").
- **Predictions come from your measurements**: a cost model of ms/token decode,
  ms/prompt-token prefill and overhead, plus the latest agent baseline. With an
  empty notebook it uses stated priors and says so.
- **Every trial is a command a human could type.** It's printed in full, with
  its `--predict` values, and its notebook rows are ordinary rows tagged
  `source=autopilot`.
- **Never touches load-time knobs** (quant, `-t`, `-ngl`, `-c`). It tells you
  when one is worth testing and leaves the restart to you.
- **Never changes two things at once.** Specs are validated to change exactly one
  setting, and agent results are checked against their baseline with `compare`'s
  multi-setting detector; a confounded result is flagged in its learning.
- **Doesn't pass lessons for you.** `progress` shows autopilot predictions
  separately (`autopilot a/b`), and only your own predictions gate lessons.

Each trial appends one record to `learnings.jsonl` next to the notebook (the last
five fields link it back to the notebook rows):

```json
{"type":"learning","ts":"2026-10-23T00:05:07Z",
 "experiment":"machine sweep --backend llama-server --template chatml --var cache-prompt --values false,true ... --predict prefill_ms=178,6.4",
 "prediction":{"prefill_ms":[177.883,6.353]},"actual":{"prefill_ms":[190.367,32.298]},"error_pct":{"prefill_ms":80.3},"passed":false,
 "learning":"Changing only cache_prompt from false to true moved prefill_ms from 190.4 ms to 32.3 ms (-83%, real difference (|t|≥2)). On a cache hit the server evaluated 1 of 28 prompt tokens. The prefill_ms@cache_prompt=true prediction missed by 80%, so the assumption behind it does not hold here.",
 "next_priority":"agent:memory_window=1: Every agent run used full history",
 "group":"grp-18e0ffcc2c67034b","key":"sweep:cache_prompt","single_variable":true,"backend":"llama-server@http://127.0.0.1:8080","simulated":false}
```

Dry run on the real notebook from [`examples/`](examples/README.md):

```
$ machine autopilot --dry-run --budget 10 --backend llama-server --template chatml
═══ AUTOPILOT (dry run — nothing will execute) ═══
backend   : llama-server@http://127.0.0.1:8080
notebook  : 96 rows from this backend
budget    : 10 min, at most 40 trials
cost model: decode 35.6 ms/token, prefill 6.35 ms/prompt token, overhead 1.3 ms (measured)
bottleneck: decode-bound: 93% of a 64-token completion is decode (35.6 ms/token)

#1  sweep:cache_prompt   lesson 3   est 5.1 s
    why : Prefix caching is the top untried lever. With an identical prompt, a cache hit should re-evaluate ~1 token: prefill 178 ms → 6.4 ms.
    $ machine sweep --backend llama-server --template chatml --var cache-prompt --values false,true --repeat 5 --warmup 1 --max-tokens 8 --ignore-eos --metric prefill_ms --lesson 3 --label ap-sweep-cache_prompt --predict prefill_ms=178,6.4

#2  agent:memory_window=1   lesson 6   est 45.4 s
    why : Every agent run used full history. Null-hypothesis prediction: a 1-line window changes nothing when episodes are ~2–3 steps. Baseline: 10 episodes, success 50%, 3.1 calls, 4538 ms/task.
    $ machine agent --backend llama-server --template chatml --memory-window 1 --lesson 6 --label ap-agent-memory_window-1 --predict success=0.50 --predict calls=3.1 --predict wall_ms=4538

#3  agent:cache_prompt   lesson 6   est 45.4 s
    why : The agent re-sends ~174 prompt tokens per call with an unchanging prefix. Prediction: prefix caching saves ~2935 ms/task (prefill of 174 → ~25 tokens per call); success and calls unchanged. Baseline: 10 episodes, success 50%, 3.1 calls, 4538 ms/task.
    $ machine agent --backend llama-server --template chatml --cache-prompt --lesson 6 --label ap-agent-cache_prompt --predict success=0.50 --predict calls=3.1 --predict wall_ms=1603

── would run 3 trial(s), ~1.6 min of the 10 min budget. Stops: no high-value single-variable experiment remains.
```

Running that plan for real took 2.2 minutes and found three things
([transcript](examples/autopilot/run.txt), [learnings](examples/autopilot/learnings.txt)):

- A prompt-cache hit evaluated 1 of 28 tokens, but prefill only fell from 190 ms
  to 32 ms. The predicted ~6 ms was wrong because prefill has a fixed per-call
  cost that a per-token model can't see.
- Prefix caching on the agent cut wall time per task from 4.5 s to 1.9 s
  (−59%), with success and calls unchanged. All three predictions held.
- A 1-line memory window *hurt*: success fell from 50% to 30% and calls rose
  from 3.1 to 3.9. The prediction of no change was wrong, and a replication
  reproduced the result.

## Scientist mode

The autopilot works through a fixed checklist. Scientist mode keeps an explicit,
inspectable **model of the system**, fitted from the notebook, and picks the
experiment that would teach that model the most per second spent. It doesn't
search over several variables at once: every pick is still one validated
single-variable `machine …` command with `--predict`, run and scored exactly
like a human's.

```sh
machine analyze                              # fit everything, print what's known / uncertain / broken (no runs)
machine beliefs                              # the explicit belief state (--json = beliefs.json)
machine beliefs --what-if "prompt_tokens=512 max_tokens=64 cache=true"   # ask the surrogate
machine scientist                            # ranked next experiments with every score component
machine scientist --loop --budget 45         # run them: the autopilot loop with the scientist's selector
machine scientist --loop --dry-run           # show the plan only
machine report --out report.md               # markdown research note
```

**What it models** (every choice below is printed by `machine analyze` / `report`):

| belief | model | why this one |
|---|---|---|
| prefill | `fixed + per_token × evaluated prompt tokens` | the cache-hit trial showed a fixed per-call cost a per-token model can't see |
| decode | `fixed + per_token × output tokens` | per-token slope is the decode speed |
| knob effects | difference of means vs the baseline setting | cache hit, temperature, quantization (your tagged runs), each agent knob |
| interaction | slope of cache-hit prefill vs cached prompt length | "does caching still pay for long prompts?" |
| agent configs | Wilson interval on success, bootstrap on calls and wall | full 10-task runs only |
| failure modes | explicit rules over episode metrics | planning error vs format loop vs tool error |

- Lines are **Theil–Sen** fits (median of pairwise slopes), which are robust to warm-up outliers.
- Intervals come from a **95% percentile bootstrap** with a fixed seed, so the same notebook always gives the same beliefs.
- Each belief is **KNOWN** (parameter interval within ±15%, or effect interval excluding 0), **≈ ZERO** (effect interval inside its practical-zero band), **UNCERTAIN**, or **UNTESTED**.

**How it picks**: `score = gain × (0.25 + 0.75·value) / (1 + log2(1 + cost_s/10))`.

- **gain** is how much the trial would shrink the target belief's interval. It's 1 for an untested belief; otherwise it is `u·(1 − √(n/(n+m)))`.
- **value** is how much that belief governs agent time or success, read off the surrogate. For example, prefill is 77% of an agent call here.
- **cost** is the surrogate's estimate of how long the trial takes.

Candidates are:

- the checklist's untried levers;
- refinements of fitted lines;
- the cache × prompt-length interaction test (`sweep --pad-to 512`);
- replications of uncertain agent effects, with priority for configs whose runs disagreed.

Below a score of 0.02 a trial is "low value" and the loop stops. With fewer than 15 run rows it falls back to the checklist.

**On the real notebook** ([`examples/scientist/`](examples/scientist/)), `machine analyze` recovers the lessons found by hand and by the autopilot:

```
── COST MODEL (95% intervals)
  prefill  ≈ 30.7 ms [27.5 ms, 37.9 ms] + 6.15 ms [5.92 ms, 6.29 ms] × evaluated prompt tokens
  decode   ≈ -32.1 ms [-45.5 ms, -13.6 ms] + 36.1 ms [35.7 ms, 36.7 ms] × output tokens
  ⇒ prefill is 77% of a baseline agent call: prompt-side levers dominate agent cost

── KNOWN TO BE ≈ ZERO
  interaction.cache_x_prompt             +0.01 ms [-0.00 ms, +0.03 ms]   a cache hit is equally cheap for long prompts

── STILL UNCERTAIN
  effect.agent.critic=model.success      +0 pts [-40 pts, +40 pts]
  effect.agent.memory_window=1.success   -20 pts [-55 pts, +15 pts]
  effect.agent.cache_prompt.wall_ms      -867 ms [-3340 ms, +3019 ms]

── BROKEN ASSUMPTIONS / ANOMALIES
  ✗ A per-token-only prefill model fails on cache hits     (34.7 ms measured vs 7.2 ms predicted)
  ✗ Decode has a negative fixed cost                       (≈ −0.9 tokens; timer likely excludes a token — untested)
  ✗ The model critic approves wrong answers                (3 false accepts, success unchanged at 50%)
  ✗ Agent wall time did not reproduce across runs of …cache=true…   (2468 vs 4894 ms/task)

── AGENT FAILURE MODES
  wrong answer (planning) 49% · format loop 32% · tool error 19%
  systematic: tasks 0, 1, 3, 8, 9 fail in every configuration — no knob fixes these
```

**Better ordering on moderate data.** On the notebook from before any autopilot run
([`01-dry-run-vs-checklist.txt`](examples/scientist/01-dry-run-vs-checklist.txt)):

- The checklist runs `cache sweep → memory window → agent cache`.
- The scientist runs `cache sweep (0.51) → agent cache (0.24) → memory window (0.18)`, because agent caching targets the 77% of each call that is prefill.

During the real loop ([`02-loop-run.txt`](examples/scientist/02-loop-run.txt), 6 trials, 3.4 min), it re-planned after the first trial and ran a test the checklist doesn't have:

1. **Cache sweep.** The fixed + per-token model's cache-hit prediction missed by 31% (measured 33 ms). The autopilot's per-token-only model had missed the same experiment by 80%.
2. **Cache × prompt-length interaction** at 432 tokens. A hit cost 37.5 ms against 3142 ms cold. Both predictions held, and the interaction is now **≈ zero**.
3. **Agent caching.** Wall time per task fell 46%.
4. **Memory window.** Wall time per task rose 52%.
5. **Agent caching, replicated.** It did *not* reproduce (+8%), even though cache hits were confirmed (31 evaluated tokens per call). The effect went from KNOWN back to UNCERTAIN, and the anomaly above now asks for another replication.
6. **Baseline, replicated.** It reproduced within 1%, so the slowdown in trial 5 isn't the baseline drifting.

`machine beliefs` prints every belief with its interval, n, status and method
([`04-beliefs.txt`](examples/scientist/04-beliefs.txt)). `--what-if` prices any
configuration with an interval and warns when extrapolating. `machine report`
writes [a research note](examples/scientist/report.md) you can check in.

**Limitations, stated:**

- The surrogate is two straight lines plus a constant. It knows nothing about context limits, batching or GPU offload, and it extrapolates poorly.
- Agent intervals rest on 10 tasks and one small model.
- Agent effects are compared unpaired, which is conservative.
- The weights in the score are judgment calls, printed every time so you can argue with them.

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
src/advisor.rs        progress gating, `next`, `wrap-up`, untried levers
src/autopilot.rs      deterministic proposer, cost model, unattended loop
src/learnings.rs      learnings.jsonl records and `machine learnings`
src/scientist/        fit (Theil–Sen, bootstrap, Wilson) · analyze · beliefs · surrogate · selector · report
src/lessons.rs        the six lesson cards
.machine/notebook.jsonl   append-only raw data (override with MACHINE_HOME)
.machine/learnings.jsonl  one interpreted record per autopilot trial
.machine/beliefs.json     latest belief state (+ beliefs_history.jsonl, one line per scientist trial)
```

## CI

`.github/workflows/ci.yml` runs fmt, clippy, tests and a simulator smoke test, then a
**real-model job**: it builds a pinned `llama-server` (static, portable CPU flags),
downloads Qwen2.5-0.5B-Instruct Q4_K_M and Q8_0 from Hugging Face, caches both
with `actions/cache`, runs [`examples/real-session.sh`](examples/real-session.sh),
then runs `machine autopilot --budget 10` and `machine scientist --loop --budget 10` on the
resulting notebook, writes the key
tables, the autopilot's learnings and the scientist's research report to the job summary, and uploads transcripts,
notebook and learnings as an artifact. Prediction misses on different hardware don't fail the build — they're data.
