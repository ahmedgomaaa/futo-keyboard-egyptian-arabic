# -*- coding: utf-8 -*-
"""Evaluate on a dataset stored as pre-computed Whisper features + token labels.

Yahya-Mohamed/egyptian-arabic-speech-dataset ships `input_features` (mel) and
`labels` (whisper token ids) instead of raw audio, so it needs its own loader.
Feature bin count is checked against the model to avoid silently scoring garbage.
"""
import os, sys, glob, re, unicodedata
import numpy as np
import torch, jiwer
import pyarrow.parquet as pq
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


def load(pdir, n, split="test"):
    files = sorted(glob.glob(os.path.join(pdir, "**", "*.parquet"), recursive=True))
    files = [f for f in files if split in os.path.basename(f)] or files
    feats, labs = [], []
    for fp in files:
        pf = pq.ParquetFile(fp)
        for b in pf.iter_batches(batch_size=32):
            d = b.to_pydict()
            for f, l in zip(d["input_features"], d["labels"]):
                feats.append(f); labs.append(l)
            if len(feats) >= n:
                return feats[:n], labs[:n]
    return feats[:n], labs[:n]


def as_array(f):
    a = np.asarray(f, dtype=np.float32)
    if a.ndim == 1:                 # flattened
        for bins in (80, 128):
            if a.size % bins == 0:
                return a.reshape(bins, -1)
    if a.ndim == 3:
        a = a[0]
    return a


def main():
    pdir, n = os.path.expanduser(sys.argv[1]), int(sys.argv[2])
    models = sys.argv[3].split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    feats, labs = load(pdir, n)
    if not feats:
        sys.exit("no rows")
    a0 = as_array(feats[0])
    print("test set: %s  (%d utterances, feature shape %s)" % (pdir, len(feats), a0.shape))

    tok = WhisperProcessor.from_pretrained("openai/whisper-small").tokenizer
    refs = []
    for l in labs:
        ids = [int(x) for x in l if int(x) >= 0]
        refs.append(norm(tok.decode(ids, skip_special_tokens=True)))

    for mid in models:
        try:
            m = WhisperForConditionalGeneration.from_pretrained(
                mid, torch_dtype=torch.float16).to(dev).eval()
            want = m.config.num_mel_bins
            if a0.shape[0] != want:
                print("  %-52s SKIP (features %d bins, model wants %d)"
                      % (mid.split('/')[-1][:52], a0.shape[0], want))
                del m; torch.cuda.empty_cache(); continue
            hyps = []
            for i in range(0, len(feats), 8):
                batch = np.stack([as_array(f) for f in feats[i:i + 8]])
                x = torch.tensor(batch, dtype=torch.float16, device=dev)
                with torch.no_grad():
                    ids = m.generate(x, language="ar", task="transcribe",
                                     max_new_tokens=180, num_beams=1)
                hyps += [norm(t) for t in tok.batch_decode(ids, skip_special_tokens=True)]
            p = [(a, b) for a, b in zip(refs, hyps) if a]
            print("  %-52s WER %6.2f  CER %6.2f  (n=%d)"
                  % (mid.split('/')[-1][:52],
                     100 * jiwer.wer([a for a, _ in p], [b for _, b in p]),
                     100 * jiwer.cer([a for a, _ in p], [b for _, b in p]), len(p)))
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print("  %-52s FAILED %s" % (mid.split('/')[-1][:52], str(e)[:70]))


if __name__ == "__main__":
    main()
