#!/usr/bin/env bash
# Reproduce the real-model session in examples/README.md.
#
# Requires a running llama-server with a Q4 model on :8080:
#   llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 2048 -t <threads> --port 8080
# Optional Q8 comparison: set LLAMA_SERVER and MODEL_Q8 / MODEL_Q4 so the script
# can restart the server between quantizations.
#
# Usage: examples/real-session.sh [path/to/machine] [out_dir]
# Predictions below are the ones written down for a 2-vCPU Xeon. On other
# hardware they will miss — that's the point: read the gap, then re-predict.
set -euo pipefail

M=${1:-./target/release/machine}
OUT=${2:-examples/out}
mkdir -p "$OUT"
export MACHINE_HOME=${MACHINE_HOME:-$OUT/home}
export MACHINE_BACKEND=llama-server MACHINE_TEMPLATE=chatml
THREADS=${THREADS:-$(nproc)}

step() { echo; echo "▶ $1"; }
wait_up() { for _ in $(seq 1 60); do curl -sf localhost:8080/health >/dev/null && return 0; sleep 1; done; echo "server did not come up" >&2; exit 1; }
restart() { # $1 = model path
  pkill -x llama-server || true; sleep 1
  "$LLAMA_SERVER" -m "$1" -c 2048 -t "$THREADS" --port 8080 > "$OUT/server.log" 2>&1 &
  wait_up
}

[ -n "${LLAMA_SERVER:-}" ] && [ -n "${MODEL_Q4:-}" ] && restart "$MODEL_Q4"
wait_up

step "env";                     "$M" env | tee "$OUT/00-env.txt"
step "lesson 1: one completion"; "$M" run --lesson 1 --max-tokens 64 --predict decode_tok_s=40 --predict wall_ms=2000 | tee "$OUT/01-first-run.txt"
step "lesson 2: noise floor";    "$M" run --lesson 2 --repeat 8 --max-tokens 64 --ignore-eos --label q4 --tag quant=q4_k_m --tag threads="$THREADS" \
                                    --predict decode_tok_s=27 --predict wall_ms=2600 | tee "$OUT/02-repeat-q4.txt"
step "lesson 3: max_tokens";     "$M" sweep --lesson 3 --var max-tokens --values 16,32,64,128 --repeat 3 --ignore-eos --quiet \
                                    --predict wall_ms=780,1340,2490,4730 | tee "$OUT/03-sweep-max-tokens.txt"
step "lesson 3: prompt length";  "$M" sweep --lesson 3 --var prompt-tokens --values 64,256,1024 --repeat 3 --max-tokens 8 --ignore-eos --quiet \
                                    --metric prefill_ms --predict prefill_ms=450,1800,7200 | tee "$OUT/03-sweep-prompt-tokens.txt"

if [ -n "${LLAMA_SERVER:-}" ] && [ -n "${MODEL_Q8:-}" ]; then
  step "lesson 3: q8 vs q4 (load-time knob → restart + tag)"
  restart "$MODEL_Q8"
  "$M" run --lesson 3 --repeat 8 --warmup 1 --max-tokens 64 --ignore-eos --label q8 --tag quant=q8_0 --tag threads="$THREADS" \
      --predict decode_tok_s=21 --predict peak_rss_mib=770 | tee "$OUT/03-repeat-q8.txt"
  "$M" compare --label q4 --label q8 --metric decode_tok_s --metric prefill_tok_s --metric wall_ms --metric peak_rss_mib | tee "$OUT/03-compare-q4-q8.txt"
  restart "$MODEL_Q4"
fi

step "lesson 4: agent loop";     "$M" agent --lesson 4 --tasks 0,4,9 --verbose --label greedy --predict calls=2 --predict wall_ms=3500 --predict success=1 | tee "$OUT/04-agent-verbose.txt"
step "lesson 5: sampling";       "$M" agent --lesson 5 --critic none --temperature 0.8 --label nocritic --predict success=0.4 --predict calls=3.5 | tee "$OUT/05-agent-nocritic.txt"
step "lesson 5: greedy";         "$M" agent --lesson 5 --critic none   --label greedy-none   --predict success=0.3  --predict calls=3.7 | tee "$OUT/05-agent-greedy-none.txt"
step "lesson 5: model critic";   "$M" agent --lesson 5 --critic model  --label greedy-critic --predict success=0.55 --predict calls=4.5 | tee "$OUT/05-agent-greedy-critic.txt"
step "lesson 5: oracle critic";  "$M" agent --lesson 5 --critic oracle --label greedy-oracle --predict success=0.7  --predict calls=4   | tee "$OUT/05-agent-greedy-oracle.txt"
"$M" compare --label greedy-none --label greedy-critic --label greedy-oracle \
    --metric success --metric calls --metric total_tokens --metric wall_ms --metric critic_ms | tee "$OUT/05-compare-critics.txt"
step "lesson 6: temperature";    "$M" sweep --lesson 6 --var temperature --values 0,0.8 --repeat 5 --max-tokens 64 --ignore-eos --quiet \
                                    --metric decode_tok_s --predict decode_tok_s=28,27 | tee "$OUT/06-sweep-temperature.txt"
step "progress + wrap-up";       "$M" progress | tee "$OUT/06-progress.txt"; "$M" wrap-up | tee "$OUT/06-wrap-up.txt"
cp "$MACHINE_HOME/notebook.jsonl" "$OUT/notebook.jsonl"
echo; echo "done → $OUT"
