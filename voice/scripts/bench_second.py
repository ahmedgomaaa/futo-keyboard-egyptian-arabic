# -*- coding: utf-8 -*-
"""Evaluate baseline vs fine-tuned models on an INDEPENDENT Egyptian test set.

Purpose: every result so far comes from Casablanca. If the baseline was trained on
Casablanca, it holds an unfair in-domain advantage and our "training makes it worse"
conclusion would be an artefact. A second set neither model was trained on
disambiguates that.
"""
import os, sys, glob, io, re, time, json, unicodedata
import numpy as np
import torch, jiwer
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
PUNCT = re.compile(u"[^ء-ي٠-٩a-zA-Z0-9\\s]")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = s.replace(u"أ", u"ا").replace(u"إ", u"ا").replace(u"آ", u"ا")
    s = s.replace(u"ى", u"ي").replace(u"ة", u"ه")
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_cols(names):
    a = next((c for c in names if c in ("audio", "original_audio", "wav", "speech")), None)
    t = next((c for c in names if c in ("transcription", "text", "sentence",
                                        "transcript", "label")), None)
    return a, t


def load(pdir, n):
    files = sorted(glob.glob(os.path.join(pdir, "**", "*.parquet"), recursive=True))
    rows = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        ac, tc = find_cols(pf.schema_arrow.names)
        if not ac or not tc:
            continue
        for b in pf.iter_batches(batch_size=64, columns=[ac, tc]):
            d = b.to_pydict()
            for a, t in zip(d[ac], d[tc]):
                tt = norm(t)
                if len(tt.split()) < 2:
                    continue
                rows.append((a, tt))
            if len(rows) >= n * 3:
                break
        if len(rows) >= n * 3:
            break
    rows.sort(key=lambda r: r[1])
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def decode(a):
    raw = a["bytes"] if isinstance(a, dict) else a
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr


def run(mid, rows, dev):
    try:
        proc = WhisperProcessor.from_pretrained(mid)
    except Exception:
        proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    m = WhisperForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.float16).to(dev).eval()
    refs, hyps = [], []
    for i in range(0, len(rows), 8):
        ch = rows[i:i + 8]
        arrs = [decode(a) for a, _ in ch]
        inp = proc(arrs, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = m.generate(inp.input_features.to(dev, torch.float16),
                             language="ar", task="transcribe",
                             max_new_tokens=180, num_beams=1)
        for (_, ref), txt in zip(ch, proc.batch_decode(ids, skip_special_tokens=True)):
            refs.append(ref); hyps.append(norm(txt))
    del m; torch.cuda.empty_cache()
    p = [(a, b) for a, b in zip(refs, hyps) if a]
    return (100 * jiwer.wer([a for a, _ in p], [b for _, b in p]),
            100 * jiwer.cer([a for a, _ in p], [b for _, b in p]), len(p))


def main():
    tdir, n = sys.argv[1], int(sys.argv[2])
    models = sys.argv[3].split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load(os.path.expanduser(tdir), n)
    print("test set: %s  (%d utterances)" % (tdir, len(rows)))
    if not rows:
        sys.exit("no usable rows")
    for mid in models:
        try:
            w, c, k = run(mid, rows, dev)
            print("  %-56s WER %6.2f  CER %6.2f  (n=%d)" % (mid.split('/')[-1][:56], w, c, k))
        except Exception as e:
            print("  %-56s FAILED %s" % (mid.split('/')[-1][:56], str(e)[:70]))


if __name__ == "__main__":
    main()
