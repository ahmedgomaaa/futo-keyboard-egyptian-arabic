# -*- coding: utf-8 -*-
"""Benchmark Egyptian-Arabic Whisper candidates on the Casablanca Egypt test split.

Reports WER/CER on normalized Arabic text plus wall-clock RTF, so model choice is
measured rather than guessed.

NOTE ON NORMALIZATION: for ASR scoring we *do* fold alef-hamza / ya-alef-maqsura /
ta-marbuta. That is the opposite of what the dictionary build does, and deliberately
so -- here we are measuring whether the model heard the right word, not whether it
guessed a particular orthographic convention.

Usage: bench_asr.py <parquet_dir> <out.json> [--n 150] [--models a,b,c] [--audio-ctx N]
"""
import sys, os, json, re, time, glob, unicodedata

import torch
import numpy as np
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_MODELS = [
    "futo-org/acft-whisper-small",                          # what FUTO ships (baseline)
    "openai/whisper-small",                                 # stock reference
    "MAdel121/whisper-small-egyptian-arabic",               # suggested
    "IbrahimAmin/code-switched-egyptian-arabic-whisper-small",
    "oddadmix/whisper-small-arabic-dialectal-v2",
    "itshamdi404/Egy_Arabic_whisper-small",
]

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
PUNCT = re.compile(u"[^ء-ي٠-٩a-zA-Z0-9\\s]")


def norm_ar(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = s.replace(u"أ", u"ا").replace(u"إ", u"ا").replace(u"آ", u"ا")
    s = s.replace(u"ى", u"ي").replace(u"ة", u"ه")
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_samples(pdir, n):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(pdir, "*.parquet")))
    if not files:
        sys.exit("no parquet in %s" % pdir)
    rows = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=64):
            d = batch.to_pydict()
            for i in range(len(d["transcription"])):
                dur = d.get("duration", [None] * len(d["transcription"]))[i]
                if dur is not None and not (0.8 <= dur <= 25.0):
                    continue
                txt = norm_ar(d["transcription"][i])
                if len(txt.split()) < 2:
                    continue
                rows.append({"audio": d["audio"][i], "ref": txt, "dur": dur})
            if len(rows) >= n * 3:
                break
        if len(rows) >= n * 3:
            break
    rows.sort(key=lambda r: r["ref"])          # deterministic
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def decode_audio(a):
    import soundfile as sf, io, librosa
    if isinstance(a, dict):
        if a.get("array") is not None:
            arr = np.asarray(a["array"], dtype=np.float32); sr = a["sampling_rate"]
        else:
            arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    else:
        arr, sr = sf.read(io.BytesIO(a), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr


def run_model(mid, samples, device, audio_ctx=None):
    t0 = time.time()
    # Some repos (e.g. futo-org/acft-whisper-small) ship weights only, no
    # tokenizer/preprocessor. ACFT does not change the tokenizer, so fall back
    # to the stock whisper-small processor.
    try:
        proc = WhisperProcessor.from_pretrained(mid)
    except Exception:
        sys.stderr.write("    [%s] no processor in repo, using openai/whisper-small" % mid + chr(10))
        proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.float16).to(device).eval()
    load_s = time.time() - t0

    hyps, refs, audio_s = [], [], 0.0
    t0 = time.time()
    BS = 8
    for i in range(0, len(samples), BS):
        chunk = samples[i:i + BS]
        arrs = [decode_audio(s["audio"]) for s in chunk]
        audio_s += sum(len(a) / 16000.0 for a in arrs)
        inp = proc(arrs, sampling_rate=16000, return_tensors="pt",
                   return_attention_mask=True)
        feats = inp.input_features.to(device, torch.float16)
        if audio_ctx:      # emulate FUTO's reduced encoder context
            feats = feats[:, :, :audio_ctx * 2]
        with torch.no_grad():
            ids = model.generate(feats, language="ar", task="transcribe",
                                 max_new_tokens=180, num_beams=1)
        for s, txt in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
            hyps.append(norm_ar(txt)); refs.append(s["ref"])
        sys.stderr.write("\r    %s: %d/%d" % (mid.split("/")[-1], len(hyps), len(samples)))
        sys.stderr.flush()
    infer_s = time.time() - t0
    sys.stderr.write("\n")

    del model
    torch.cuda.empty_cache()
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    return {
        "model": mid,
        "wer": round(100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]), 2),
        "cer": round(100 * jiwer.cer([r for r, _ in pairs], [h for _, h in pairs]), 2),
        "n": len(pairs), "load_s": round(load_s, 1),
        "infer_s": round(infer_s, 1), "audio_s": round(audio_s, 1),
        "rtf": round(infer_s / audio_s, 3) if audio_s else None,
        "audio_ctx": audio_ctx,
        "samples": [{"ref": r, "hyp": h} for r, h in pairs[:3]],
    }


def main():
    pdir, outp = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 250
    models = (sys.argv[sys.argv.index("--models") + 1].split(",")
              if "--models" in sys.argv else DEFAULT_MODELS)
    actx = int(sys.argv[sys.argv.index("--audio-ctx") + 1]) if "--audio-ctx" in sys.argv else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s | audio_ctx: %s" % (device, actx or "full (1500)"))
    samples = load_samples(pdir, n)
    print("evaluating on %d Casablanca-Egypt utterances (%.1f min audio)\n"
          % (len(samples), sum(s["dur"] or 0 for s in samples) / 60.0))

    results = []
    for mid in models:
        try:
            r = run_model(mid, samples, device, actx)
            results.append(r)
            print("  %-52s WER %6.2f  CER %6.2f  RTF %s" % (mid, r["wer"], r["cer"], r["rtf"]))
        except Exception as e:
            print("  %-52s FAILED: %s" % (mid, str(e)[:120]))
            results.append({"model": mid, "error": str(e)[:300]})

    results.sort(key=lambda r: r.get("wer", 999))
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"n": len(samples), "audio_ctx": actx, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("\n=== ranking (lower WER better) ===")
    for r in results:
        if "wer" in r:
            print("  %6.2f  %s" % (r["wer"], r["model"]))


if __name__ == "__main__":
    main()
