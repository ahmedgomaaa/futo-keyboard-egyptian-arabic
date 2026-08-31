set -u
D=$HOME/egdict/asr/ft_overnight
LATEST=$(ls -d "$D"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "latest checkpoint dir: $LATEST"
if [ -z "$LATEST" ]; then
  echo "NO CHECKPOINTS FOUND"
  exit 1
fi
ls -la "$LATEST"
echo "--- trainer_state.json ---"
python3 -c "
import json
d = json.load(open('$LATEST/trainer_state.json'))
print('global_step:', d['global_step'], '| epoch:', d['epoch'])
"
echo "--- safetensors integrity (can it be opened?) ---"
python3 -c "
from safetensors import safe_open
with safe_open('$LATEST/model.safetensors', framework='pt') as f:
    keys = list(f.keys())
print('OK, %d tensors' % len(keys))
"
