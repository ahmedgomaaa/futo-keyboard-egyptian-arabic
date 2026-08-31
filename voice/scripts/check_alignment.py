# -*- coding: utf-8 -*-
"""Verify harvested (audio, text) pairs are actually aligned.

Transcribes sampled segments with our shipped GGML model and compares to the caption
text. We expect moderate disagreement (two different ASR systems), but if alignment
were broken the WER would be ~100% and the transcripts would be unrelated.
"""
import json, os, re, subprocess, sys, unicodedata, random
import jiwer

ROOT = os.path.expanduser("~/egdict")
CLI = os.path.join(ROOT, "tools/whisper.cpp/build/bin/whisper-cli")
MODEL = os.path.join(ROOT, "asr/ggml/ggml-egyptian-small-q5_1.bin")
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/egdict/asr/yt_test")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
PUNCT = re.compile(u"[^ء-ي٠-٩\\s]")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = s.replace(u"أ", u"ا").replace(u"إ", u"ا").replace(u"آ", u"ا")
    s = s.replace(u"ى", u"ي").replace(u"ة", u"ه")
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


rows = [json.loads(l) for l in open(os.path.join(OUT, "manifest.jsonl"), encoding="utf-8")]
random.Random(0).shuffle(rows)
rows = rows[:N]

refs, hyps = [], []
for r in rows:
    wav = os.path.join(OUT, "wav", r["wav"])
    out = subprocess.run([CLI, "-m", MODEL, "-f", wav, "-l", "ar", "-nt", "-np", "-t", "8"],
                         capture_output=True, text=True).stdout
    refs.append(norm(r["text"]))
    hyps.append(norm(out))

pairs = [(a, b) for a, b in zip(refs, hyps) if a and b]
print("compared %d segments" % len(pairs))
print("WER caption-vs-ourmodel : %.1f%%" % (100 * jiwer.wer([a for a, _ in pairs],
                                                            [b for _, b in pairs])))
print("CER caption-vs-ourmodel : %.1f%%" % (100 * jiwer.cer([a for a, _ in pairs],
                                                            [b for _, b in pairs])))
print("\n--- side by side (first 5) ---")
for a, b in pairs[:5]:
    print("  CAPTION : %s" % a[:100])
    print("  OURMODEL: %s" % b[:100])
    print()
print("Interpretation: 25-60%% => aligned (normal ASR disagreement).")
print("                >85%%    => segmentation/timing is broken.")
