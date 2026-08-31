# Egyptian Arabic for FUTO Keyboard

Everyday Egyptian Arabic — مش، عايز، ازيك، دلوقتي، ايوه — gets flagged as typos and
mistranscribed by voice input on every Arabic keyboard, because they're all built and
tested on Modern Standard Arabic (MSA). This project fixes that for
[FUTO Keyboard](https://keyboard.futo.org), an open-source, offline-first Android
keyboard, by adding:

1. **A dictionary** so Egyptian words are recognized and autocorrect stops fighting them.
2. **A voice input model** fine-tuned on real spoken Egyptian, so dictation actually
   understands the dialect instead of transcribing it as broken MSA.

Both are free, offline, and take a few minutes to install. If you speak Egyptian
Arabic and use FUTO Keyboard, this is for you.

## Results

**Dictionary:** the stock Arabic dictionary is missing `عايز`, `ايه`, `ازاي`,
`دلوقتي`, `لسه`, `كتير`, `ازيك`, `ايوه`, `علشان` — completely absent, so they're
always flagged as typos. This dictionary adds 1.2M+ words including all of them,
pinned so autocorrect doesn't fight them.

**Voice input** — word error rate (WER) on real spoken Egyptian, lower is better,
measured on independent test sets the model never trained on:

| Model | WER | Relative improvement |
|---|---|---|
| FUTO's stock model | ~65-79 | — |
| Best public Egyptian Whisper model (unmodified) | 37.62 | 43% |
| **This project's model** | **19.99** | **72%** |

Full methodology, all the dead ends, and every number: [`voice/README.md`](voice/README.md).

## Install

**You only need the two files below — everything else in this repo is source code
and build scripts for anyone who wants to reproduce or extend the work.**

1. Download the two files from the [latest release](../../releases/latest):
   - `main_ar_eg.dict` (dictionary)
   - `ggml-egyptian-phase3-q5_1.bin` (voice model)
2. Follow [`docs/INSTALL.md`](docs/INSTALL.md) — five minutes, no technical
   knowledge needed.

## What's in this repo

| Path | What |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | Plain-language setup guide |
| [`dictionary/`](dictionary/) | Dictionary source, build scripts, corpus stats |
| [`voice/`](voice/) | Voice model training pipeline, benchmarks, full write-up |
| [releases](../../releases) | The actual `.dict` and `.bin` files to download |

## License and provenance

The code in this repo (build scripts, training pipeline) is [MIT-licensed](LICENSE).
The data and models are assembled from several sources with their own terms —
being upfront about this matters more than a blanket license statement:

- **Dictionary base**: Helium314's AOSP Arabic wordlist, itself built from
  [Leipzig Corpora](https://wortschatz.uni-leipzig.de/) data under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Egyptian frequency data**: EFC-mini and FineWeb-2 (`arz`), both public research
  corpora — see `dictionary/STATS.md` for exact sources and sizes.
- **Voice model base**: `itshamdi404/Egy_Arabic_whisper-small` on Hugging Face
  (itself a fine-tune of OpenAI's Whisper, MIT-licensed).
- **Voice training data**: MGB-3 (public broadcast speech corpus) plus audio and
  auto-generated captions harvested from public Egyptian YouTube podcasts and
  channels for training purposes. **Downloading YouTube content is against
  YouTube's Terms of Service, and the source podcasts are copyrighted.** This
  project does not redistribute any harvested audio or video — only the resulting
  model weights (a statistical artifact, not a copy of the source material), the
  same way every openly-published speech model is built. If you rerun the harvest
  scripts yourself, that download is your own responsibility under YouTube's ToS
  and applicable copyright law in your jurisdiction.

If you're a rights holder for any of the source podcasts and have concerns, please
open an issue.
