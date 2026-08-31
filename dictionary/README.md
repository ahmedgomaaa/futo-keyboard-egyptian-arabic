# Arabic (Egyptian) dictionary for FUTO Keyboard

`main_ar_eg.dict` — an AOSP-format binary dictionary covering **Modern Standard Arabic
plus Egyptian colloquial Arabic**, so everyday Egyptian words (مش، كده، عايز، ازيك،
دلوقتي، ايوه) are no longer flagged as typos.

**1,195,947 words · 308,446 bigrams · 10.37 MB**

Built by merging Helium314's MSA Arabic wordlist with a frequency list derived from
490 M tokens of Egyptian-dialect text (EFC-mini forum corpus + FineWeb-2 `arz`).
See `STATS.md` for corpora, counts, filter survival rates, and the `f` histogram.

## Install (FUTO Keyboard)

1. Copy `main_ar_eg.dict` to your phone (Downloads is fine).
2. Open **FUTO Keyboard → Languages & Models**.
3. Under **العربية**, tap **Dictionary**.
4. Choose the import / file-picker entry and select `main_ar_eg.dict`.
5. Confirm. It replaces the current main Arabic dictionary.

The file declares `locale=ar` (not `ar_EG`) on purpose, so it attaches to the existing
**العربية** entry instead of creating a separate language.

> **Warning — an emoji dictionary will overwrite this one.** FUTO/HeliBoard keeps a
> single main dictionary slot per locale. If you later import an emoji dictionary for
> Arabic, it **replaces** this file and the Egyptian vocabulary is gone. Re-import
> `main_ar_eg.dict` to restore it.

## Checking it worked

Type `دلوقتي` or `ازيك` — neither should be underlined as a typo. `عايز` and `مش` should
be offered as suggestions rather than autocorrected away. Typing `مش` should suggest
`عارف` / `فاهم` / `عايز` as next words.

## What this changes vs. the stock MSA dictionary

| | Stock `main_ar` | This dictionary |
|---|---|---|
| `عايز`, `ايه`, `ازاي`, `دلوقتي`, `لسه`, `كتير`, `ازيك`, `ايوه`, `علشان` | **absent** (flagged as typos) | present, floored at f=250 |
| `برضو` / `اهو` / `كده` | f=19 / 28 / 68 — loses to autocorrect | f=250 |
| Egyptian spellings `فى`, `اللى`, `دى` | absent | present |
| Entries with embedded tatweel (`فـي`, `المـلـف`) | 16,470 defective entries | repaired or dropped |
| Next-word prediction | MSA only | Egyptian bigrams from forum text |

175 core Egyptian words are pinned at `f=250` so autocorrect never fights them; the
full list with corpus counts is in `floor_words.tsv`. 20 sexual/offensive terms that
occur frequently in the corpus are included but marked `possibly_offensive=true`, so
they are suppressed when offensive-word blocking is on.

## Rebuilding

`build.sh` reproduces everything end to end — downloads, filtering, merge, compile,
validation, and `STATS.md`. It needs a JRE 8+ and Python 3.8+ with `pyarrow`:

```sh
./build.sh                      # builds into ~/egdict
ROOT=/tmp/egdict ./build.sh     # or somewhere else
INCLUDE_OPENSUBS=1 ./build.sh   # fold in OpenSubtitles (not recommended, see STATS.md 1.1)
```

Re-running with corpora already downloaded takes a few minutes; the first run has to
fetch ~3 GB.

## Files

| File | What it is |
|---|---|
| `main_ar_eg.dict` | the importable binary dictionary |
| `ar_eg.combined` | the plain-text source wordlist it was compiled from |
| `floor_words.tsv` | the 175 pinned core-Egyptian words, with corpus counts |
| `build.sh` + `scripts/` | reproducible end-to-end build |
| `STATS.md` | corpora, token counts, filter survival, `f` histogram, dropped words |

## Known limitations

- **`dicttool_aosp.jar` cannot decompile `.dict` files.** That direction needs a native
  JNI library that upstream does not ship, so the usual round-trip check is impossible.
  Instead the toolchain is verified by rebuilding the official published `main_ar.dict`
  byte-for-byte from its published source, and the compiled binary's header is parsed
  and asserted directly. See `STATS.md` §7.
- The dictionary includes 741,396 Egyptian words seen ≥5 times in the corpus. Some are
  genuine misspellings that cross that threshold; raising `NEW_WORD_BAR` in `build.sh`
  trades vocabulary coverage for stricter typo correction.
- Bigrams come from forum and web text, so next-word predictions carry that register
  (football and religion are well represented).
