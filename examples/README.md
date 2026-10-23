# A real session: Qwen2.5-0.5B on a 2-vCPU box

Everything on this page is a real measurement from one sitting on 2026-10-22,
not the simulator. The raw rows are in
[`notebook-2vcpu-qwen0.5b.jsonl`](notebook-2vcpu-qwen0.5b.jsonl) and the full
terminal output of every step is in [`transcripts/`](transcripts/).
The predictions were written down before each command ran (they're the
`--predict` flags), using only numbers from earlier steps.

| | |
|---|---|
| machine | Intel Xeon @ 2.80GHz, 2 vCPU, 7.8 GiB RAM, no GPU, Ubuntu 24.04 |
| backend | llama.cpp `llama-server` @ `184f653`, `-c 2048 -t 2` |
| model | Qwen2.5-0.5B-Instruct, GGUF Q4_K_M (469 MiB) and Q8_0 (644 MiB) |
| template | `--template chatml` (Qwen's chat format) |

Reproduce it with [`real-session.sh`](real-session.sh) — or just push: the
`real-model` CI job builds the same llama-server, downloads both GGUFs (cached
after the first run), runs the script on the GitHub runner and attaches the
transcripts + notebook as a build artifact. Your numbers will differ; the
*shape* of the results is what should transfer.

---

## Lesson 1 — one completion ([01-first-run](transcripts/01-first-run.txt))

```
predicted decode 40 tok/s, wall 2000 ms
#1  prefill 28 tok 234.9 ms | decode 35 tok 1323.6 ms = 26.4 t/s | wall 1563.5 ms | ovh 5.1 ms | rss 587.9→591.4 MiB
decode_tok_s  error 51%  ✗      wall_ms  error 28%  ✗
```

Both predictions missed. Reading the line explains why:

- **decode dominates**: 1324 ms of 1563 ms. Prefill of 28 tokens was 235 ms.
- **the model stopped at 35 tokens, not 64** — end-of-sequence. wall_ms was
  measuring *how long the answer was*, not decode speed. The tool now flags this
  and suggests `--ignore-eos`; every timing run after this uses it.
- overhead (HTTP + JSON) was 5 ms — negligible.

Re-predicted from that line (≈ 28 tok/s, so ≈ 36 ms/token):
[32 tokens](transcripts/01-rerun-32.txt) predicted 1340 ms → **1322 ms (1.4% off)**,
[128 tokens](transcripts/01-rerun-128.txt) predicted 4760 ms → **4834 ms (1.5% off)**.

## Lesson 2 — noise floor ([02-repeat-q4](transcripts/02-repeat-q4.txt))

8 identical runs, 64 tokens, greedy:

```
decode_tok_s   mean 28.1   sd 0.68   cv 2.4%
prefill_ms     mean 208    sd 16.2   cv 7.8%
distinct outputs: 1/8
```

Decode cv 2.4% means a ~5% speed change is the smallest you can trust here.
Prefill is noisier (7.8%) because 28 tokens is ~200 ms of work — scheduler
jitter is a bigger share. Greedy output was identical all 8 times.

## Lesson 3 — single-variable sweeps

**Output length** ([sweep](transcripts/03-sweep-max-tokens.txt)) — predictions from lesson 1's per-token cost:

```
max_tokens   16      32      64      128
wall_ms      746     1335    2534    4764      (all four predictions within 5%)
fit: wall_ms ≈ 194 + 35.9 × max_tokens   R² = 1.000
```

35.9 ms per output token, ~194 ms fixed (prefill + overhead). Perfectly linear.

**Prompt length** ([sweep](transcripts/03-sweep-prompt-tokens.txt)) — I assumed prefill scales like the 28-token run (≈ 7 ms/token):

```
measured prompt_tokens   57     207     852
prefill_ms               375    1228    5373     (2 of 3 predictions off by 34–47%)
fit: prefill_ms ≈ −31 + 5.26 × prompt_tokens   R² = 0.999
```

The miss was real learning: prefill gets *cheaper per token* as the prompt
grows (6.6 → 5.9 → 6.3 ms/token vs 7.4 for 28 tokens), because larger batches
amortise per-call work. Also, the padding target (1024) produced 852 real
tokens — the tokenizer decides, so the fit uses the server's count.

**Quantization** — a load-time knob: restart the server, tag, compare
([q8 run](transcripts/03-repeat-q8.txt), [compare](transcripts/03-compare-q4-q8.txt)):

```
metric           q4_k_m       q8_0
decode_tok_s     28.1±0.68    24.9±0.51   −11.3%  real difference
prefill_tok_s    135±9.7      148±8.9     +9.6%   real difference
peak_rss_mib     592          737         +24.4%  real difference
```

Q8 moves ~1.4× the bytes per weight, so memory-bound decode got slower and RSS
grew. Prefill (compute-bound) got slightly *faster* — less dequantisation work
per weight. One knob, opposite effects on the two phases.

## Lesson 4 — the agent loop ([verbose trace](transcripts/04-agent-verbose.txt))

```
[✗] task 0  answer 288 (truth 2016)   calls 2
    actor 146+10 tok 1209 ms → Calc("12*24")
    actor 161+8  tok 1252 ms → Answer(288)        ← computed 12×24, forgot ×7
[✓] task 4  answer 696                calls 3
[✗] task 9  answer none (truth 13)    calls 6     ← replied bare "13" three times: format errors
```

I predicted 2 calls/task and 100% success; reality was 3.7 calls and 33%.
Each call costs ~1.2–2 s, almost all of it **prefill of a ~150–260 token prompt**
that grows with every history line. The 0.5B model's failures are format
(bare numbers) and planning (dropping a factor), not arithmetic — the tool
does the arithmetic.

After rebuilding this lesson smaller (single tasks, `wrap-up`'s suggestion),
calls and wall time were predicted within 6% ([task 0](transcripts/04-rebuild-task0.txt), [task 4](transcripts/04-rebuild-task4.txt)).

## Lesson 5 — does a critic pay for itself? ([compare](transcripts/05-compare-critics.txt))

All 10 tasks, greedy:

```
                 none        model critic   oracle
success          50%         50%            50%
calls/task       3.1         3.9            4.2
wall_ms/task     4538        4979           6486
critic false accepts: 3, false rejects: 0
```

- The model critic took 11% of the agent's time (+0.8 calls/task) and bought
  **zero** accuracy: it said YES to all 3 wrong answers it saw (the other 2
  failures never produced an answer — format errors until `--max-steps`).
- Even the oracle (perfect verification, free) didn't help: on rejection the
  greedy model re-derives the same wrong plan until `--max-steps`. The
  bottleneck is the actor's planning, not verification.
- Sampling at temperature 0.8 ([run](transcripts/05-agent-nocritic.txt)) dropped
  accuracy from 50% to **0%** while decode speed was unchanged
  ([temperature sweep](transcripts/06-sweep-temperature.txt): 28.3 vs 28.8 tok/s, within noise).

`compare` also caught a confounded comparison I made: putting the
temperature-0.8 run next to the greedy critic runs prints
`⚠ differs … in more than one setting`.

## Lesson 6 — closing the loop ([progress](transcripts/06-progress.txt), [wrap-up](transcripts/06-wrap-up.txt))

All six lessons passed (≥2 predictions within ±20%, latest included). The
weakest idea was still the agent (mean prediction error 32%), and the advisor's
untried next step, grounded in the ~150-token instruction prefix every call
re-sends:

```
WHAT SHOULD I BUILD NEXT?
  Prompt-prefix caching hasn't been measured. With a fixed prompt, a cache hit should collapse prefill_ms.
  $ machine sweep --backend llama-server --lesson 3 --var cache-prompt --values false,true --metric prefill_ms --repeat 5
```

Then the same switch on the agent (`machine agent --cache-prompt`): since
prefill is roughly three quarters of each agent call here, prefix caching is the largest
untested lever on this box.

## Autopilot on the same notebook ([`autopilot/`](autopilot/))

`machine autopilot --budget 10` read this notebook, saw that every lever except
prefix caching and the memory window had been tried, and ran 5 trials in 2.2 min
([dry run](autopilot/dry-run.txt), [run](autopilot/run.txt),
[learnings](autopilot/learnings.txt), [raw](autopilot/learnings.jsonl)). The
headline: prefix caching cut the agent's wall time per task by 59% (4.5 s → 1.9 s)
at the same accuracy.

## Scientist mode on the same data ([`scientist/`](scientist/))

`machine scientist --loop --budget 20`, started from the notebook above (before
the autopilot), ran 6 trials in 3.4 min and kept an explicit belief state the
whole way ([run](scientist/02-loop-run.txt), [beliefs](scientist/04-beliefs.txt),
[report](scientist/report.md)). It established that a cache hit costs the same
~35 ms fixed prefill whether the cached prompt is 28 or 432 tokens, and it
caught that the agent-caching speed-up did not reproduce on a second run, so
it moved that effect back to UNCERTAIN instead of reporting it as known.
