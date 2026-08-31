# -*- coding: utf-8 -*-
"""Diagnose the CER anomaly: is the combined model hallucinating on some utterances?

WER improving while CER doubles is the signature of occasional runaway output
(repetition loops / hallucinated continuations). Find the worst offenders and look
at them, rather than shipping on average metrics alone.
"""
import os, sys, glob, io, re, unicodedata
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


D = os.path.expanduser("~/egdict/asr/test2/gomaa1k")
f = sorted(glob.glob(os.path.join(D, "**", "*.parquet"), recursive=True))[0]
pf = pq.ParquetFile(f)
rows = []
for b in pf.iter_batches(batch_size=64, columns=["audio", "text"]):
    d = b.to_pydict()
    for a, t in zip(d["audio"], d["text"]):
        tt = norm(t)
        if len(tt.split()) >= 2:
            rows.append((a, tt))
    if len(rows) >= 600:
        break
rows.sort(key=lambda r: r[1])
rows = rows[::max(1, len(rows) // 200)][:200]

dev = "cuda"
MODELS = {
    "combined": os.path.expanduser("~/egdict/asr/ft_combo/final"),
    "baseline": "itshamdi404/Egy_Arabic_whisper-small",
}

res = {}
for tag, mid in MODELS.items():
    try:
        proc = WhisperProcessor.from_pretrained(mid)
    except Exception:
        proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    m = WhisperForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.float16).to(dev).eval()
    hyps = []
    for i in range(0, len(rows), 8):
        ch = rows[i:i + 8]
        arrs = []
        for a, _ in ch:
            raw = a["bytes"] if isinstance(a, dict) else a
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            arrs.append(arr)
        inp = proc(arrs, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = m.generate(inp.input_features.to(dev, torch.float16),
                             language="ar", task="transcribe",
                             max_new_tokens=180, num_beams=1)
        hyps += [norm(t) for t in proc.batch_decode(ids, skip_special_tokens=True)]
    res[tag] = hyps
    del m; torch.cuda.empty_cache()

refs = [r for _, r in rows]
print("%-10s %8s %8s  %s" % ("model", "WER", "CER", "len-ratio hyp/ref"))
for tag, hyps in res.items():
    lr = sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs))
    print("%-10s %8.2f %8.2f  %.2f" % (tag, 100 * jiwer.wer(refs, hyps),
                                       100 * jiwer.cer(refs, hyps), lr))

print("\n--- worst CER cases for the combined model ---")
scored = []
for i, (ref, hyp) in enumerate(zip(refs, res["combined"])):
    if not ref:
        continue
    c = jiwer.cer(ref, hyp) if hyp else 1.0
    scored.append((c, i, ref, hyp))
scored.sort(reverse=True)
runaway = sum(1 for c, i, r, h in scored if len(h) > 3 * max(len(r), 1))
print("utterances where hypothesis > 3x reference length: %d / %d\n" % (runaway, len(scored)))
for c, i, r, h in scored[:4]:
    print("  CER %.0f%%" % (100 * c))
    print("    REF (%3d ch): %s" % (len(r), r[:110]))
    print("    HYP (%3d ch): %s" % (len(h), h[:110]))
    print()
