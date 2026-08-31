#!/usr/bin/env bash
# Resume from training onward. combo_v2 (34,692 utterances) is already built and
# verified on disk from the previous attempt -- reuse it rather than rebuilding.
set -uo pipefail

ROOT=$HOME/egdict
S=/mnt/c/Users/Ahmed/Egyptian-dic/scripts
LOG=$ROOT/work/overnight3.log
: > "$LOG"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

TARGET_EPOCHS=3
MIN_STEPS=800
MAX_STEPS=7000

N_ROWS=$(wc -l < "$ROOT/asr/combo_v2/manifest.jsonl" 2>/dev/null || echo 0)
if [ "$N_ROWS" -lt 1000 ]; then
  log "combo_v2 missing or too small ($N_ROWS rows), rebuilding"
  rm -rf "$ROOT/asr/combo_v2"
  python3 -u "$S/build_combo.py" --out "$ROOT/asr/combo_v2" \
    --yt-dir "$ROOT/asr/yt" --mgb-dir "$ROOT/asr/train/mgb3" >> "$LOG" 2>&1
  N_ROWS=$(wc -l < "$ROOT/asr/combo_v2/manifest.jsonl")
fi
# User wants this wrapped up in <=2h total. Cut the step target rather than the
# epoch-based formula: at the observed ~1.8s/step, current progress (~1520) plus
# ~3000 more steps is ~1.5h train + ~0.5h convert/eval, fitting the budget.
STEPS=${OVERNIGHT_STEPS_OVERRIDE:-$(python3 -c "
n=$N_ROWS; eb=16; ep=$TARGET_EPOCHS
s=int(n/eb*ep)
print(max($MIN_STEPS, min($MAX_STEPS, s)))
")}
log "=== PART 2 START: reusing combo_v2 ($N_ROWS utterances) -> $STEPS steps ==="

log "--- patching any existing checkpoint to the fine-grained interval ---"
# HF Trainer stores save_steps/eval_steps inside checkpoint-*/trainer_state.json and
# keeps honoring that stored value on resume, silently overriding --ckpt-every. Without
# this the last several resumes were quietly reverting to the original 1084-step
# interval, which is why nothing checkpointed between step 1084 and 2168.
python3 - "$ROOT/asr/ft_overnight" >> "$LOG" 2>&1 <<'PY'
import glob, json, os, sys
d = sys.argv[1]
ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
if ckpts:
    p = os.path.join(ckpts[-1], "trainer_state.json")
    j = json.load(open(p, encoding="utf-8"))
    j["save_steps"] = 80
    j["eval_steps"] = 80
    json.dump(j, open(p, "w", encoding="utf-8"))
    print("patched %s -> save_steps=eval_steps=80" % p)
else:
    print("no checkpoint to patch yet")
PY

# Kill any stray trainer from a prior invocation. Harness "killed" notifications on
# this box have NOT reliably meant the underlying WSL process actually died -- the
# training script was piping all output into $LOG and producing near-zero stdout,
# which appears to make the task-tracking give up and report "killed" while the
# real process kept training headless. Two of these ran concurrently on the same
# GPU/checkpoint dir for ~20 min tonight before being caught. Guard against it here.
if pkill -9 -f "train_whisper.py.*ft_overnight" 2>/dev/null; then
  log "!! killed a still-running trainer from a previous invocation"
  sleep 3
fi

log "--- training ---"
# tee to real stdout (not just $LOG) so the harness sees periodic live output and
# doesn't mistake this for a hung/dead task.
PYTHONUNBUFFERED=1 python3 -u "$S/train_whisper.py" \
  --train-dir "$ROOT/asr/combo_v2" \
  --dev-dir   "$ROOT/asr/train/casa_val" \
  --base itshamdi404/Egy_Arabic_whisper-small \
  --out "$ROOT/asr/ft_overnight" \
  --steps "$STEPS" --bs 8 --accum 2 --lr 1e-5 --max-dev 200 --ckpt-every 80 \
  2>&1 | tee -a "$LOG"
log "training finished rc=$?"
if [ ! -d "$ROOT/asr/ft_overnight/final" ]; then
  log "!! no final checkpoint, aborting"
  log "=== PART 2 END (FAILED) ==="
  exit 1
fi

log "--- convert to ggml + quantize ---"
SRC="$ROOT/asr/ft_overnight/final"
python3 - "$SRC" >> "$LOG" 2>&1 <<'PY'
import os, sys
if not os.path.exists(os.path.join(sys.argv[1], "vocab.json")):
    from transformers import WhisperProcessor
    WhisperProcessor.from_pretrained("itshamdi404/Egy_Arabic_whisper-small").save_pretrained(sys.argv[1])
    for cache in [os.path.expanduser("~/.cache/huggingface/hub/models--itshamdi404--Egy_Arabic_whisper-small"),
                  os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-small")]:
        import glob, shutil
        v = glob.glob(os.path.join(cache, "**", "vocab.json"), recursive=True)
        if v:
            d = os.path.dirname(v[0])
            for f in ("vocab.json", "added_tokens.json", "merges.txt"):
                p = os.path.join(d, f)
                if os.path.exists(p):
                    shutil.copy(p, sys.argv[1])
            break
print("processor files ready")
PY
cd "$ROOT/tools"
python3 whisper.cpp/models/convert-h5-to-ggml.py "$SRC" "$ROOT/tools/whisper" "$ROOT/asr/ggml" >> "$LOG" 2>&1
mv -f "$ROOT/asr/ggml/ggml-model.bin" "$ROOT/asr/ggml/ggml-egyptian-overnight-f16.bin"
"$ROOT/tools/whisper.cpp/build/bin/whisper-quantize" \
  "$ROOT/asr/ggml/ggml-egyptian-overnight-f16.bin" \
  "$ROOT/asr/ggml/ggml-egyptian-overnight-q5_1.bin" q5_1 >> "$LOG" 2>&1
log "ggml files ready"

log "--- evaluation ---"
python3 -u "$S/bench_second.py" "$ROOT/asr/test2/gomaa1k" 200 \
  "$SRC,itshamdi404/Egy_Arabic_whisper-small" >> "$LOG" 2>&1
python3 -u "$S/bench_features.py" "$ROOT/asr/test2/yahya" 300 \
  "$SRC,itshamdi404/Egy_Arabic_whisper-small" >> "$LOG" 2>&1
python3 -u "$S/eval_ggml_clean.py" 150 \
  "$ROOT/asr/ggml/ggml-egyptian-overnight-q5_1.bin,$ROOT/asr/ggml/ggml-egyptian-combo-q5_1.bin,$ROOT/asr/ggml/ggml-egyptian-small-q5_1.bin" >> "$LOG" 2>&1

log "=== PART 2 END: ALL DONE ==="
{
  echo "=== OVERNIGHT SUMMARY ==="
  echo "combined training utterances: $N_ROWS"
  echo "steps trained: $STEPS"
  echo
  grep -A20 "evaluation ---" "$LOG" | tail -30
} > "$ROOT/work/OVERNIGHT_SUMMARY.txt"
