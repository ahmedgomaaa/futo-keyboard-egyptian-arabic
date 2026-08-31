# -*- coding: utf-8 -*-
"""Rebuild the combined training manifest from scratch: MGB-3 + ALL harvested YouTube data.

Run again any time the YouTube harvest grows -- it always pulls the current full
manifest.jsonl under --yt-dir, so re-running after more harvesting picks up everything.

Usage: build_combo.py --out ~/egdict/asr/combo --yt-dir ~/egdict/asr/yt --mgb-dir ~/egdict/asr/train/mgb3
"""
import argparse, json, os, sys, glob, io
import soundfile as sf
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_whisper import clean_ref, find_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--yt-dir", required=True)
    ap.add_argument("--mgb-dir", required=True)
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    wav_out = os.path.join(out, "wav")
    os.makedirs(wav_out, exist_ok=True)
    mf = open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8")
    n = 0

    yt_mf = os.path.join(os.path.expanduser(a.yt_dir), "manifest.jsonl")
    if os.path.exists(yt_mf):
        for line in open(yt_mf, encoding="utf-8"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            p = os.path.join(os.path.expanduser(a.yt_dir), "wav", j["wav"])
            if os.path.exists(p) and j.get("text"):
                mf.write(json.dumps({"wav_abs": p, "text": j["text"]}, ensure_ascii=False) + "\n")
                n += 1
    print("youtube rows: %d" % n)
    yt_n = n

    mgb = os.path.expanduser(a.mgb_dir)
    i = 0
    for f in sorted(glob.glob(os.path.join(mgb, "**", "*.parquet"), recursive=True)):
        pf = pq.ParquetFile(f)
        ac, tc = find_cols(pf.schema_arrow.names)
        if not ac or not tc:
            continue
        for b in pf.iter_batches(batch_size=32, columns=[ac, tc]):
            d = b.to_pydict()
            for aud, t in zip(d[ac], d[tc]):
                t = clean_ref(t)
                if len(t.split()) < 2:
                    continue
                raw = aud["bytes"] if isinstance(aud, dict) else aud
                try:
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                except Exception:
                    continue
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                p = os.path.join(wav_out, "mgb%06d.wav" % i)
                i += 1
                sf.write(p, arr, sr, subtype="PCM_16")
                mf.write(json.dumps({"wav_abs": p, "text": t}, ensure_ascii=False) + "\n")
                n += 1
    mf.close()
    print("mgb3 rows: %d" % (n - yt_n))
    print("TOTAL combined manifest rows: %d" % n)


if __name__ == "__main__":
    main()
