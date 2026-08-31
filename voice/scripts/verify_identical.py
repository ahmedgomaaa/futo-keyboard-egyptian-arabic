# -*- coding: utf-8 -*-
"""Check whether MAdel121/whisper-small-egyptian-arabic differs from stock whisper-small."""
import torch
from transformers import WhisperForConditionalGeneration

A = "openai/whisper-small"
B = "MAdel121/whisper-small-egyptian-arabic"

a = WhisperForConditionalGeneration.from_pretrained(A, torch_dtype=torch.float32)
b = WhisperForConditionalGeneration.from_pretrained(B, torch_dtype=torch.float32)

sa, sb = a.state_dict(), b.state_dict()
print("tensors: %d vs %d" % (len(sa), len(sb)))
only_a = set(sa) - set(sb)
only_b = set(sb) - set(sa)
if only_a or only_b:
    print("  keys only in A:", list(only_a)[:5])
    print("  keys only in B:", list(only_b)[:5])

identical, differing, max_absdiff = 0, [], 0.0
for k in sorted(set(sa) & set(sb)):
    ta, tb = sa[k], sb[k]
    if ta.shape != tb.shape:
        differing.append((k, "shape %s vs %s" % (ta.shape, tb.shape)))
        continue
    if torch.equal(ta, tb):
        identical += 1
    else:
        d = (ta - tb).abs().max().item()
        max_absdiff = max(max_absdiff, d)
        differing.append((k, "max|diff|=%.3e" % d))

print("\nidentical tensors : %d" % identical)
print("differing tensors : %d" % len(differing))
print("largest abs diff  : %.3e" % max_absdiff)
if differing:
    print("\nfirst differing tensors:")
    for k, why in differing[:10]:
        print("   %-60s %s" % (k, why))
else:
    print("\n==> The two checkpoints are BIT-IDENTICAL: this 'finetune' is stock whisper-small.")
