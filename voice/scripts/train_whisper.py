# -*- coding: utf-8 -*-
"""Fine-tune Whisper on Egyptian Arabic (MGB-3), starting from the best checkpoint.

Sized for a single 8 GB GPU: fp16 + gradient checkpointing + 8-bit Adam if available.

IMPORTANT: trains on MGB-3 and evaluates on the Casablanca Egypt *validation* split.
The Casablanca Egypt *test* split is never touched here so it stays a clean held-out
benchmark for comparing against the models we already measured.

Usage:
  train_whisper.py --train-dir ~/egdict/asr/train/mgb3 \
                   --dev-dir   ~/egdict/asr/train/casa_val \
                   --base itshamdi404/Egy_Arabic_whisper-small \
                   --out ~/egdict/asr/ft --steps 3000
"""
import os, sys, glob, io, re, json, argparse, unicodedata, random

import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")


def clean_ref(s):
    """Light cleanup only. Unlike scoring we keep orthography as written -- the model
    should learn the spelling conventions people actually use."""
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = re.sub(r"\[[^\]]{0,30}\]", " ", s)     # [Music], [Applause] ...
    s = re.sub(r"<[^>]{0,30}>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_cols(names):
    # Prefer the unprocessed audio: MGB-3 also ships a source-separated variant,
    # but training on enhanced audio mismatches what a phone mic actually feeds us.
    for cand in ("audio", "original_audio", "wav", "speech", "separated_target_audio"):
        if cand in names:
            audio = cand
            break
    else:
        audio = None
    text = next((c for c in names if c in
                 ("transcription", "text", "sentence", "transcript", "label")), None)
    return audio, text


def load_manifest(pdir, limit=None):
    """Harvested YouTube data: manifest.jsonl + wav/ directory."""
    mf = os.path.join(pdir, "manifest.jsonl")
    rows = []
    with open(mf, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            t = clean_ref(j.get("text"))
            if not t or len(t.split()) < 2:
                continue
            # "wav_abs" lets a combined manifest reference files from several corpora
            wav = j.get("wav_abs") or os.path.join(pdir, "wav", j["wav"])
            if os.path.exists(wav):
                rows.append((wav, t))
            if limit and len(rows) >= limit:
                break
    return rows


def load_rows(pdir, limit=None, split_hint=None):
    if os.path.exists(os.path.join(pdir, "manifest.jsonl")):
        return load_manifest(pdir, limit)
    files = sorted(glob.glob(os.path.join(pdir, "**", "*.parquet"), recursive=True))
    if split_hint:
        files = [f for f in files if split_hint in os.path.basename(f)] or files
    rows = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        acol, tcol = find_cols(pf.schema_arrow.names)
        if not acol or not tcol:
            continue
        for b in pf.iter_batches(batch_size=64, columns=[acol, tcol]):
            d = b.to_pydict()
            for a, t in zip(d[acol], d[tcol]):
                t = clean_ref(t)
                if not t or len(t.split()) < 2:
                    continue
                rows.append((a, t))
            if limit and len(rows) >= limit:
                return rows[:limit]
    return rows


class SpeechDS(Dataset):
    def __init__(self, rows, processor, max_s=30.0):
        self.rows, self.p, self.max_s = rows, processor, max_s

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, text = self.rows[i]
        if isinstance(a, str):                      # harvested wav on disk
            arr, sr = sf.read(a, dtype="float32")
        else:
            raw = a["bytes"] if isinstance(a, dict) else a
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        arr = arr[: int(self.max_s * 16000)]
        feats = self.p.feature_extractor(arr, sampling_rate=16000,
                                         return_tensors="np").input_features[0]
        labels = self.p.tokenizer(text, truncation=True, max_length=200).input_ids
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, processor):
        self.p = processor

    def __call__(self, batch):
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        lab = self.p.tokenizer.pad({"input_ids": [b["labels"] for b in batch]},
                                   return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        # strip the leading BOS the model adds back itself
        if (labels[:, 0] == self.p.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dev-dir")
    ap.add_argument("--base", default="itshamdi404/Egy_Arabic_whisper-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-dev", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="absolute step interval for save/eval; overrides steps//24")
    a = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(a.base, language="ar", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    train_rows = load_rows(a.train_dir, limit=a.max_train, split_hint="train")
    print("train utterances: %d" % len(train_rows))
    dev_rows = load_rows(a.dev_dir, limit=a.max_dev) if a.dev_dir else []
    print("dev utterances  : %d" % len(dev_rows))
    if not train_rows:
        sys.exit("no training data found under %s" % a.train_dir)

    random.Random(0).shuffle(train_rows)
    tds = SpeechDS(train_rows, proc)
    dds = SpeechDS(dev_rows, proc) if dev_rows else None

    try:
        import bitsandbytes  # noqa
        optim = "adamw_bnb_8bit"
    except Exception:
        optim = "adafactor"
    print("optimizer:", optim)

    args = Seq2SeqTrainingArguments(
        output_dir=a.out,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        warmup_steps=int(0.05 * a.steps),
        max_steps=a.steps,
        gradient_checkpointing=True,
        fp16=True,
        optim=optim,
        logging_steps=25,
        # Finer-grained checkpoints: this box keeps getting suspended (interval
        # has ranged from 3 to 90 minutes, cause unclear), so checkpoint often
        # enough that no single interruption costs much progress.
        save_steps=a.ckpt_every or max(1, a.steps // 24),
        save_total_limit=2,
        eval_strategy="steps" if dds else "no",
        eval_steps=a.ckpt_every or max(1, a.steps // 24),
        save_strategy="steps" if dds else "no",
        per_device_eval_batch_size=4,
        predict_with_generate=False,
        # 13.6h is a small set for Whisper -- keep the checkpoint that actually
        # generalises to the dev split rather than the last (likely overfit) one.
        load_best_model_at_end=bool(dds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tds,
                             eval_dataset=dds, data_collator=Collator(proc))

    # Auto-resume: if a checkpoint already exists in --out (e.g. the process was
    # killed mid-run), continue from it instead of restarting at step 0.
    ckpts = sorted(glob.glob(os.path.join(a.out, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    if resume:
        print("resuming from checkpoint: %s" % resume)
    trainer.train(resume_from_checkpoint=resume)

    final = os.path.join(a.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    proc.save_pretrained(final)
    print("saved:", final)


if __name__ == "__main__":
    main()
