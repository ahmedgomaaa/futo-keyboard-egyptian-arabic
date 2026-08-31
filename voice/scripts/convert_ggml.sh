set -e
MODEL="itshamdi404/Egy_Arabic_whisper-small"
ROOT=$HOME/egdict
SRC=$ROOT/asr/hf/Egy_Arabic_whisper-small
OUT=$ROOT/asr/ggml
mkdir -p "$SRC" "$OUT"

echo "== 1. snapshot the HF model to a plain directory =="
python3 - "$MODEL" "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
mid, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=mid, local_dir=dest)
print("snapshot at:", p)
PY
ls -la "$SRC"

echo "== 2. ensure tokenizer/preprocessor files exist (converter needs them) =="
python3 - "$SRC" <<'PY'
import sys, os, json
d = sys.argv[1]
need = ["vocab.json", "tokenizer_config.json", "config.json"]
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
print("missing:", missing or "none")
cfg = json.load(open(os.path.join(d, "config.json")))
print("model_type:", cfg.get("model_type"), "| d_model:", cfg.get("d_model"),
      "| enc layers:", cfg.get("encoder_layers"), "| vocab:", cfg.get("vocab_size"))
PY

echo "== 3. convert HF -> ggml =="
cd $ROOT/tools
python3 whisper.cpp/models/convert-h5-to-ggml.py "$SRC" "$ROOT/tools/whisper" "$OUT"
ls -l "$OUT"
