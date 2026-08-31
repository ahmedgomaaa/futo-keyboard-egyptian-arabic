# STATS - Arabic (Egyptian) dictionary build

## 1. Corpora

| Corpus | Source | On disk | Raw Arabic tokens | Kept tokens | Unique types | Used? |
|---|---|---|---|---|---|---|
| EFC-mini | `faisalq/EFC-mini` (HF) | 1.9 GB | 192,108,103 | 190,716,638 | 2,219,929 | yes |
| FineWeb-2 `arz` | `HuggingFaceFW/fineweb-2` (HF) | 1.1 GB | 299,553,017 | 299,303,843 | 3,121,609 | yes |
| OpenSubtitles ar | OPUS v2018 mono | 1.01 GB | 361,275,819 | 361,104,783 | 2,002,635 | **no - see 1.1** |
| **Used total** | | **3.0 GB** | **491,661,120** | **490,020,481** | **4,218,555** | |

**Not used:** `Mostafanofal453/2.5-Million-Rows-Egyptian-Datasets-Collection` - the GitHub repo
contains only a README; the data lives on Kaggle behind authentication. Skipped per instructions.

**Not used:** `arz` Wikipedia - excluded by instruction (bot-generated template translation).

### 1.1 Why OpenSubtitles was excluded

It was specified as "conversational, heavy Egyptian dub content". It is not - it is
overwhelmingly MSA fansub translation. Rate of Egyptian markers per million tokens:

| Marker | EFC | FineWeb | OpenSubtitles |
|---|---|---|---|
| مش | 5995.2 | 2763.9 | 14.0 |
| اللى | 2903.7 | 1332.2 | 3.2 |
| كده | 1708.3 | 941.5 | 5.1 |
| علشان | 912.9 | 646.2 | 1.7 |
| ازيك | 267.8 | 12.3 | 0.2 |
| **all 12 markers** | **15611.9** | **7604.6** | **36.4** |

Its top-20 tokens are textbook MSA. At 361M tokens it would have been the largest input
(42% of the total). Because the count-to-`f` calibration is fitted on vocabulary shared with
the MSA base, including it would inflate MSA counts while leaving Egyptian counts untouched,
pushing Egyptian words *relatively lower*. It is downloaded and counted; re-enable by adding
`counts_opensubs.tsv` to the `combine_counts.py` call in `build.sh`.

## 2. Filter survival

| Corpus | Raw tokens | `too_long` (>20 ch) | `elongation` (3+ same letter) | Survival |
|---|---|---|---|---|
| EFC-mini | 192,108,103 | 14,299 | 1,377,166 | 99.28% |
| FineWeb-2 arz | 299,553,017 | 20,698 | 228,476 | 99.92% |
| OpenSubtitles* | 361,275,819 | 116,262 | 54,774 | 99.95% |

\* counted but not merged.

Examples dropped by each rule:

- **EFC-mini / too_long:** الإمبراطوريةالأهلاوية، مشكورمشكورمشكورمشكورمشكور، مشكورمشكورمشكورمشكورمشكورمشكورمشكورمشكورمشكورمشكور، لالالالالالالالالالالا، لالالالالالالالالالالالا، لالالالالالالالالالالالالا
- **EFC-mini / elongation:** ههههه، خخخخخ، مبروووووك، جدااااا، كووورة، يااااارب، ياااااه، اوووووه
- **FineWeb-2 arz / too_long:** هههههههههههههههههههههه، ههههههههههههههههههههه، ههههههههههههههههههههههه، هههههههههههههههههههههههه، ههههههههههههههههههههههههه، هههههههههههههههههههههههههههه
- **FineWeb-2 arz / elongation:** جدااا، ههههه، هههههههه، هههههه، هههه، ههههههه، كتييير، ههههههههه

A **minimum-count filter of 5** was then applied to the combined list:

- types kept: **1,055,862**
- types dropped (seen 1-4 times): **3,162,693**

## 3. Base wordlist repair

The MSA base ships with defective entries: embedded tatweel, hyphens, and Arabic
presentation-form ligatures.

| Action | Count |
|---|---|
| entries normalized (NFKC + tatweel/diacritic strip) | 12,786 |
| entries dropped (not Arabic after normalization) | 3,684 |
| base entries before -> after | 470,783 -> 454,551 |

Repairs: بـ->ب، لـ->ل، هـ->ه، كـ->ك، ومواطـــنات->ومواطنات، بالـ->بال

Dropped: ـ، ــ، أيلول-سبتمبر، الأوسط-وكالات، شباط-فبراير، يونيو-حزيران

## 4. Merge

Corpus counts are mapped to `f` by a least-squares fit of the base's `f` on `log(count)`
over the shared vocabulary, so Egyptian `f` values land on the same scale as the MSA base
rather than an invented one:

```
f = 12.9175 * ln(count) + -16.3112     (clipped to 1..254)
fitted on 314,466 shared words, R^2 = 0.544
```

| Outcome | Count |
|---|---|
| words raised above their base `f` by corpus evidence | 158,465 |
| words where the base `f` was already higher (kept) | 156,001 |
| new Egyptian words added | 741,396 |
| core Egyptian words floored at f=250 | 175 |
| words marked `possibly_offensive=true` | 20 |
| **final word entries** | **1,195,947** |
| **final bigrams** | **308,219** |

## 5. `f` distribution

```
  0- 15 | ######################################################## 520,572
 16- 31 | ###################################                      331,239
 32- 47 | #################                                        161,743
 48- 63 | #########                                                85,962
 64- 79 | ####                                                     46,390
 80- 95 | ##                                                       24,904
 96-111 | #                                                        13,006
112-127 | #                                                        6,711
128-143 | #                                                        3,139
144-159 | #                                                        1,409
160-175 | #                                                        505
176-191 | #                                                        144
192-207 | #                                                        33
208-223 | #                                                        10
224-239 | #                                                        3
240-255 | #                                                        177
```

## 6. Floor list

175 curated core-Egyptian words pinned at `f=250`, 103 of them absent from the MSA base entirely.

Selection is **curated, not derived**. Ranking candidates by "frequent in corpus but missing
from the base" was tried first and rejected: on Egyptian forum text it surfaces football nouns,
proper nouns, a concatenation artifact, and porn spam - none of which belong at f=250. The
blocklist in `derive_floor.py` holds those out.

- **Category A** (129): Egyptian core function / discourse words
- **Category B** (46): Egyptian orthographic variants (hamza-less alef, ya/alef-maqsura)

Full list with corpus counts is in `floor_words.tsv`.

## 7. Validation

| Check | Result |
|---|---|
| toolchain reproduces official `main_ar.dict` byte-for-byte | PASS (md5 `f08701717e66613ed51d7fd111f346d7`) |
| `locale` is plain `ar` | PASS |
| `version` > 18 | PASS (54) |
| no duplicate entries | PASS (0) |
| all `f` in 0..255 | PASS |
| all words pure Arabic script | PASS |
| every bigram target exists as a word | PASS (0 orphans) |
| binary magic `0x9BC13AFE` | PASS |
| binary header attributes intact | PASS |
| 30 Egyptian test words present | PASS |
| 31 MSA test words present | PASS |
| 16 non-floored Egyptian words present with calibrated `f` | PASS |

**Final size: 10.37 MB** (budget 15 MB).

> `dicttool_aosp.jar` cannot decompile `.dict` -> `.combined`: that direction needs a native
> JNI library (`latinime-aosp-dicttool-host`) that is not shipped, and it fails on both Linux
> and Windows. The round-trip check was therefore replaced by (a) reproducing the official
> published dictionary byte-for-byte from its published source, and (b) parsing the compiled
> binary's header directly.
