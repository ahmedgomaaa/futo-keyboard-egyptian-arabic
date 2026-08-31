set -e
ROOT=$HOME/egdict
W=$ROOT/tools/whisper.cpp
OUT=$ROOT/asr/ggml
QB=$W/build/bin/whisper-quantize

echo "== quantize =="
for Q in q5_1 q8_0; do
  "$QB" "$OUT/ggml-egyptian-small-f16.bin" "$OUT/ggml-egyptian-small-$Q.bin" $Q 2>&1 | tail -2
done
ls -l "$OUT"

echo
echo "== extract a real Egyptian test clip from Casablanca =="
python3 - <<'PY'
import glob, io, os
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
d = os.path.expanduser('~/egdict/asr/casablanca_egypt')
f = sorted(glob.glob(os.path.join(d, '*.parquet')))[0]
pf = pq.ParquetFile(f)
b = next(pf.iter_batches(batch_size=40)).to_pydict()
outdir = os.path.expanduser('~/egdict/asr/clips')
os.makedirs(outdir, exist_ok=True)
picked = 0
for i in range(len(b['transcription'])):
    if (b['duration'][i] or 0) < 3.0:
        continue
    arr, sr = sf.read(io.BytesIO(b['audio'][i]['bytes']), dtype='float32')
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    sf.write(os.path.join(outdir, 'clip%d.wav' % picked), arr, 16000, subtype='PCM_16')
    print('clip%d.wav  REF: %s' % (picked, b['transcription'][i]))
    picked += 1
    if picked >= 3:
        break
PY
