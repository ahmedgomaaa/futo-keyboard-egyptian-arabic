# Egyptian Arabic voice input for FUTO Keyboard

A Whisper model fine-tuned on Egyptian Arabic, converted to GGML for FUTO Keyboard's
voice input. It replaces FUTO's stock multilingual model, which is close to unusable
for spoken Egyptian dialect.

**`ggml-egyptian-phase3-q5_1.bin` — 190 MB — install this one.**

## Install

1. Copy `ggml-egyptian-phase3-q5_1.bin` to your phone.
2. FUTO Keyboard → **Settings → Voice Input**.
3. Add / import a model, select the `.bin`.
4. Set it as the active model for Arabic.

## Results

Measured through **whisper.cpp on the quantized file** — the same engine FUTO runs —
on `gomaa1k`, an Egyptian test set none of these models were trained on. All four
scored on the same 150 utterances in one run, so the comparison is apples-to-apples:

| Model | WER | CER | repetition loops |
|---|---|---|---|
| **ggml-egyptian-phase3-q5_1** (shipped) | **19.99** | **7.73** | 0/150 |
| ggml-egyptian-overnight-q5_1 (previous) | 24.54 | 9.93 | 0/150 |
| ggml-egyptian-combo-q5_1 (earlier) | 28.62 | 11.30 | 0/150 |
| ggml-egyptian-small-q5_1 (original best public model) | 37.62 | 14.86 | 0/150 |

**47% relative error reduction over the original public model, 18.5% over the
previous in-house model.** Confirmed on two other independent test sets too (PyTorch
checkpoint, not just the quantized file): gomaa1k 27.74 vs 35.61, yahya 39.28 vs 44.03
— consistent improvement, not measurement noise. For reference, FUTO's stock
`acft-whisper-small` scores ~65-79 WER depending on benchmark (see "the benchmark
trap" below for why that range is wide).

## How it was built

1. Benchmarked 8 public Egyptian/Arabic Whisper models; `itshamdi404/Egy_Arabic_whisper-small`
   won and became the base.
2. First fine-tune: 25.6 h (MGB-3 broadcast + 12 h harvested YouTube podcasts) -> the
   `combo` model, 26.76 WER.
3. Second fine-tune (overnight run): harvested **40 more hours** from a much broader
   sweep of Egyptian channels (86 channels touched, 18 search queries), for **65.4 h
   total** (MGB-3 13.6h + YouTube ~51.8h). Continued training from the `combo`
   checkpoint for ~1.7 epochs over the enlarged set -> `overnight`, 24.54 WER.
4. Converted to GGML and quantized to q5_1 at each stage.

`build`: `collect_ids.py` -> `harvest.py` -> `build_combo.py` -> `train_whisper.py`
-> `convert-h5-to-ggml.py` -> `whisper-quantize`, evaluated with `eval_ggml_clean.py`.

More data continued to help with no sign of diminishing returns yet (26.76 -> 24.54
from +40h). Scaling further is the obvious next step if it's worth another overnight run.

## Things that were tried and did NOT work

### A bigger model is worse

`whisper-large-v3-turbo` has 3.3x the parameters and would be ~570 MB quantized:

| Model | Size | WER (Casablanca) |
|---|---|---|
| Egy_Arabic_whisper-small | 190 MB | **46.63** |
| whisper-large-v3-turbo-egyptian | ~570 MB | 47.47 |
| openai/whisper-large-v3-turbo | ~570 MB | 51.04 |

Turbo gets *better* CER but *worse* WER: it hears the phonetics right, then writes them
with MSA orthography, so whole words score wrong. Capacity does not fix a dialect
spelling mismatch.

### `MAdel121/whisper-small-egyptian-arabic` is not a finetune

Its weights are bit-identical to `openai/whisper-small` — all 480 tensors, maximum
absolute difference exactly 0.0 (verified with `verify_identical.py`). Installing it
changes nothing.

## The benchmark trap (important if you continue this work)

Early on, three separate fine-tuning runs all appeared to make the model **worse**,
and the work was nearly abandoned on that basis. The cause was the benchmark, not
the training: the base model had evidently been trained on Casablanca, so measuring
on Casablanca flattered it and punished any further training.

| Test set | base model | after finetuning |
|---|---|---|
| Casablanca (contaminated) | 46.63 | 53.47 |
| gomaa1k (independent) | 40.28 | **37.34** |
| yahya (independent) | 45.22 | **42.16** |

**Never evaluate on Casablanca for this work.** Use `gomaa1k` / `yahya`, or another
set the base model has not seen. Three consistent results in the same direction were
a reason to doubt the ruler, not the data.

A second measurement trap: PyTorch greedy decoding made the finetuned model look
barely better (37.34 vs 40.28) because it fell into repetition loops on ~2% of
utterances. whisper.cpp's temperature fallback eliminates those, and the real gap on
the shipped artifact is 26.76 vs 38.63. **Always measure the quantized `.bin` through
whisper.cpp**, not the HF checkpoint.

## Training on an unstable machine (read this before an unattended run)

The overnight training run for this model was interrupted **10+ times** by the host
box going down (WSL2 restarts, and separately, the training process dying while WSL
itself stayed up -- two different failure modes, cause never fully confirmed). Total
elapsed wall-clock for a ~1.5h training job was over 24 hours. Two things made this
survivable instead of fatal:

1. **`--ckpt-every N`** in `train_whisper.py` -- checkpoint far more often than the
   default (every ~1/24th of total steps). At the default interval an interruption
   could cost a full epoch; at `--ckpt-every 80` it costs about 2 minutes.
2. **Auto-resume** -- `train_whisper.py` automatically resumes from the latest
   `checkpoint-*` in `--out` if one exists. No flag needed, just rerun the same command.

One sharp edge if you hit it: **HF's `Trainer` bakes `save_steps`/`eval_steps` into
`checkpoint-*/trainer_state.json` and keeps honoring that stored value on resume**,
silently overriding a new `--ckpt-every` passed on the command line (only a warning is
logged, easy to miss). If you change the checkpoint interval mid-run, patch the stored
value directly before resuming:

```python
import json
p = "checkpoint-N/trainer_state.json"
j = json.load(open(p))
j["save_steps"] = j["eval_steps"] = 80   # your new interval
json.dump(j, open(p, "w"))
```

`overnight_part2.sh` does this automatically on every launch.

The other trap: **a "killed" notification from the task runner did not reliably mean
the training process actually died.** The training script piped all output into a log
file with nothing going to real stdout for long stretches; something in the process
supervision appears to treat prolonged stdout silence as a hung task and reports it
"killed" while the real process keeps training, orphaned, on the GPU. Twice this
produced **two full trainers running concurrently on the same 8GB GPU, writing
checkpoints to the same directory** -- before that was caught, it manifested as
inference slowing from ~1.8s/step to ~3.4s/step. Fix: `overnight_part2.sh` now (a)
pipes training output through `tee` so the harness sees continuous stdout, and (b)
force-kills any process matching the training command line before launching a new one,
every single time, regardless of what a prior notification claimed. **Always verify
actual process state (`ps aux | grep train_whisper`, `nvidia-smi`) before relaunching
after an interruption -- never trust the notification alone.**

## Harvesting more data

The pipeline is reusable. It's already been run twice (12h, then +40h more from a
broader channel/query sweep -- 65.4h total on disk now), and there's no sign yet that
more data stops helping:

```sh
python3 scripts/collect_ids.py 20 8        # gather video ids across channels
python3 scripts/harvest.py --ids-file ~/egdict/work/harvest_ids.txt \
        --out ~/egdict/asr/yt --max-hours 40   # additional hours, not a hard total
python3 scripts/check_alignment.py ~/egdict/asr/yt 15   # verify before training
python3 scripts/build_combo.py --out ~/egdict/asr/combo_v2 \
        --yt-dir ~/egdict/asr/yt --mgb-dir ~/egdict/asr/train/mgb3
```

`harvest.py`'s `--max-hours` adds that many *new* hours on top of whatever is already
in `--out` (it skips videos already in the manifest), so re-running with a bigger
channel/query sweep each time keeps compounding the dataset.

Initial verified-Egyptian channels, scored by dialect markers per 1k words (the
second harvest swept ~80 more channels without individually re-scoring each one --
`score_channels.py` / `check_daheeh.py` show the method if you want to vet new ones):

| Channel | eg/1k |
|---|---|
| بودكاست الدويتو | 95.7 |
| حان الآن | 94.8 |
| GLASSROOM جلاس روم | 93.4 |
| STUDIO 77 (فايق و رايق) | 91.7 |
| TPP Network | 90.1 |
| عبدالرحمن مجدي | 84.2 |
| Atheer - أثير | 82.0 |
| بودكاست بدون ورق | 76.2 |
| الدحيح | 69.1 |
| #ABtalks | no captions |

YouTube `ar-orig` auto-captions preserve Egyptian faithfully (~136 markers/1k) and use
Egyptian orthography (`جديده`, `النهارده`, `ماتعبش`), and their timings align with the
audio — verified by transcribing harvested segments and comparing (`check_alignment.py`).
They are ASR output, so they carry Google's errors; that is weak supervision, which is
how Whisper itself was trained.

Note: downloading YouTube content is against their ToS and podcasts are copyrighted.
Fine for a private keyboard model, a different question if redistributed.

## Files

| File | What |
|---|---|
| `ggml-egyptian-overnight-q5_1.bin` | **install this** — 190 MB |
| `ggml-egyptian-overnight-f16.bin` | unquantized, 488 MB |
| `ggml-egyptian-combo-q5_1.bin` | previous model (25.6h data), kept for comparison |
| `ggml-egyptian-small-q5_1.bin` | original best public model, kept for comparison |
| `scripts/` | harvest, train, convert, evaluate |

## Limitations

- Tuned for Egyptian; MSA and other dialects will be worse than a general Arabic model.
- whisper-small is the ceiling FUTO recommends for phones.
- `gomaa1k` is 150-200 utterances; the exact percentages carry sampling noise. Trust
  the direction (each retrain has beaten the last) more than the decimal.
- Real-world phone dictation was never measured directly — test it on your own voice.
- The final training run only completed ~1.7 of a planned 3 epochs over the enlarged
  65.4h dataset (cut short to hit a time budget). A full 3-epoch pass over this much
  data, or another data-scaling round, is the most likely way to improve further.
