# -*- coding: utf-8 -*-
"""WER of the shipped GGML artifacts on an independent Egyptian test set.

This is the number that actually matters: the quantized .bin, decoded by whisper.cpp
(the same engine FUTO uses), on a benchmark the baseline was not trained on.
"""
import os, sys, glob, io, re, subprocess, tempfile, unicodedata
import pyarrow.parquet as pq
import soundfile as sf
import jiwer

ROOT = os.path.expanduser("~/egdict")
CLI = os.path.join(ROOT, "tools/whisper.cpp/build/bin/whisper-cli")
D = os.path.expanduser("~/egdict/asr/test2/gomaa1k")

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
PUNCT = re.compile(u"[^ء-ي٠-٩\\s]")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = s.replace(u"أ", u"ا").replace(u"إ", u"ا").replace(u"آ", u"ا")
    s = s.replace(u"ى", u"ي").replace(u"ة", u"ه")
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
models = sys.argv[2].split(",")

f = sorted(glob.glob(os.path.join(D, "**", "*.parquet"), recursive=True))[0]
pf = pq.ParquetFile(f)
rows = []
for b in pf.iter_batches(batch_size=64, columns=["audio", "text"]):
    d = b.to_pydict()
    for a, t in zip(d["audio"], d["text"]):
        tt = norm(t)
        if len(tt.split()) >= 2:
            rows.append((a, tt))
    if len(rows) >= N * 3:
        break
rows.sort(key=lambda r: r[1])
rows = rows[::max(1, len(rows) // N)][:N]
print("test set: gomaa1k (independent), %d utterances\n" % len(rows))

tmp = tempfile.mkdtemp()
wavs = []
for i, (a, ref) in enumerate(rows):
    raw = a["bytes"] if isinstance(a, dict) else a
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    p = os.path.join(tmp, "%04d.wav" % i)
    sf.write(p, arr, 16000, subtype="PCM_16")
    wavs.append((p, ref))

for m in models:
    path = os.path.expanduser(m)
    refs, hyps, loops = [], [], 0
    for p, ref in wavs:
        out = subprocess.run([CLI, "-m", path, "-f", p, "-l", "ar", "-nt", "-np", "-t", "8"],
                             capture_output=True, text=True).stdout
        h = norm(out)
        if re.search(r"(.)\1{9,}", h):
            loops += 1
        refs.append(ref); hyps.append(h)
    pr = [(a, b) for a, b in zip(refs, hyps) if a and b]
    print("  %-34s WER %6.2f  CER %6.2f  loops %d/%d"
          % (os.path.basename(path), 100 * jiwer.wer([a for a, _ in pr], [b for _, b in pr]),
             100 * jiwer.cer([a for a, _ in pr], [b for _, b in pr]), loops, len(wavs)))
