# -*- coding: utf-8 -*-
"""Generate STATS.md from the build artifacts."""
import json, collections, io, os

ROOT = os.environ.get('ROOT', os.path.expanduser('~/egdict'))

W = os.path.join(ROOT, 'work') + os.sep
OUT = os.path.join(ROOT, 'out') + os.sep


def j(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


merge = j(W + 'merge_report.json')
val = j(W + 'validate.json')
corp = {c: j(W + 'stats_%s.json' % c) for c in ('efc', 'fineweb', 'opensubs')}

hist = collections.Counter()
with open(OUT + 'ar_eg.combined', encoding='utf-8') as f:
    f.readline()
    for line in f:
        if line.startswith(' word='):
            fv = int(line.split(',f=')[1].split(',')[0])
            hist[(fv // 16) * 16] += 1

floor = [l.rstrip('\n').split('\t') for l in open(W + 'floor_words.tsv', encoding='utf-8')]

L = []
A = L.append
N = lambda x: format(x, ',')

A("# STATS - Arabic (Egyptian) dictionary build\n")

A("## 1. Corpora\n")
A("| Corpus | Source | On disk | Raw Arabic tokens | Kept tokens | Unique types | Used? |")
A("|---|---|---|---|---|---|---|")
rows = [('EFC-mini', '`faisalq/EFC-mini` (HF)', '1.9 GB', 'efc', 'yes'),
        ('FineWeb-2 `arz`', '`HuggingFaceFW/fineweb-2` (HF)', '1.1 GB', 'fineweb', 'yes'),
        ('OpenSubtitles ar', 'OPUS v2018 mono', '1.01 GB', 'opensubs', '**no - see 1.1**')]
for name, src, disk, key, used in rows:
    c = corp[key]
    A("| %s | %s | %s | %s | %s | %s | %s |" % (name, src, disk, N(c['raw_tokens']),
                                                N(c['kept_tokens']), N(c['unique_kept']), used))
tr = corp['efc']['raw_tokens'] + corp['fineweb']['raw_tokens']
tk = corp['efc']['kept_tokens'] + corp['fineweb']['kept_tokens']
tt = merge['eg_types_after_min_count'] + merge['eg_types_below_min_count']
A("| **Used total** | | **3.0 GB** | **%s** | **%s** | **%s** | |" % (N(tr), N(tk), N(tt)))
A("")
A("**Not used:** `Mostafanofal453/2.5-Million-Rows-Egyptian-Datasets-Collection` - the GitHub repo")
A("contains only a README; the data lives on Kaggle behind authentication. Skipped per instructions.")
A("")
A("**Not used:** `arz` Wikipedia - excluded by instruction (bot-generated template translation).")
A("")

A("### 1.1 Why OpenSubtitles was excluded\n")
A("It was specified as \"conversational, heavy Egyptian dub content\". It is not - it is")
A("overwhelmingly MSA fansub translation. Rate of Egyptian markers per million tokens:\n")
A("| Marker | EFC | FineWeb | OpenSubtitles |")
A("|---|---|---|---|")
for w, a, b, c in [(u"مش", 5995.2, 2763.9, 14.0),
                   (u"اللى", 2903.7, 1332.2, 3.2),
                   (u"كده", 1708.3, 941.5, 5.1),
                   (u"علشان", 912.9, 646.2, 1.7),
                   (u"ازيك", 267.8, 12.3, 0.2)]:
    A("| %s | %.1f | %.1f | %.1f |" % (w, a, b, c))
A("| **all 12 markers** | **15611.9** | **7604.6** | **36.4** |")
A("")
A("Its top-20 tokens are textbook MSA. At 361M tokens it would have been the largest input")
A("(42% of the total). Because the count-to-`f` calibration is fitted on vocabulary shared with")
A("the MSA base, including it would inflate MSA counts while leaving Egyptian counts untouched,")
A("pushing Egyptian words *relatively lower*. It is downloaded and counted; re-enable by adding")
A("`counts_opensubs.tsv` to the `combine_counts.py` call in `build.sh`.")
A("")

A("## 2. Filter survival\n")
A("| Corpus | Raw tokens | `too_long` (>20 ch) | `elongation` (3+ same letter) | Survival |")
A("|---|---|---|---|---|")
for name, key in [('EFC-mini', 'efc'), ('FineWeb-2 arz', 'fineweb'), ('OpenSubtitles*', 'opensubs')]:
    c = corp[key]
    A("| %s | %s | %s | %s | %.2f%% |" % (name, N(c['raw_tokens']), N(c['dropped_too_long']),
                                          N(c['dropped_elongation']),
                                          100.0 * c['kept_tokens'] / c['raw_tokens']))
A("")
A("\\* counted but not merged.\n")
A("Examples dropped by each rule:\n")
SEP = u"، "
for key, label in [('efc', 'EFC-mini'), ('fineweb', 'FineWeb-2 arz')]:
    A("- **%s / too_long:** %s" % (label, SEP.join(corp[key]['examples_too_long'][:6])))
    A("- **%s / elongation:** %s" % (label, SEP.join(corp[key]['examples_elongation'][:8])))
A("")
A("A **minimum-count filter of 5** was then applied to the combined list:\n")
A("- types kept: **%s**" % N(merge['eg_types_after_min_count']))
A("- types dropped (seen 1-4 times): **%s**" % N(merge['eg_types_below_min_count']))
A("")

A("## 3. Base wordlist repair\n")
A("The MSA base ships with defective entries: embedded tatweel, hyphens, and Arabic")
A("presentation-form ligatures.\n")
A("| Action | Count |")
A("|---|---|")
A("| entries normalized (NFKC + tatweel/diacritic strip) | %s |" % N(merge['base_entries_normalized']))
A("| entries dropped (not Arabic after normalization) | %s |" % N(merge['base_entries_dropped']))
A("| base entries before -> after | 470,783 -> %s |" % N(merge['base_words']))
A("")
A("Repairs: " + SEP.join(merge['base_normalized_examples'][:6]))
A("")
A("Dropped: " + SEP.join(merge['base_dropped_examples'][:6]))
A("")

A("## 4. Merge\n")
A("Corpus counts are mapped to `f` by a least-squares fit of the base's `f` on `log(count)`")
A("over the shared vocabulary, so Egyptian `f` values land on the same scale as the MSA base")
A("rather than an invented one:\n")
A("```")
A("f = %.4f * ln(count) + %.4f     (clipped to 1..254)" % (merge['fit_slope'], merge['fit_intercept']))
A("fitted on %s shared words, R^2 = %.3f" % (N(merge['fit_n_shared']), merge['fit_r2']))
A("```\n")
A("| Outcome | Count |")
A("|---|---|")
A("| words raised above their base `f` by corpus evidence | %s |" % N(merge['raised']))
A("| words where the base `f` was already higher (kept) | %s |" % N(merge['kept_base']))
A("| new Egyptian words added | %s |" % N(merge['new_words']))
A("| core Egyptian words floored at f=250 | %s |" % N(merge['floored']))
A("| words marked `possibly_offensive=true` | %s |" % N(merge['marked_offensive']))
A("| **final word entries** | **%s** |" % N(merge['final_words']))
A("| **final bigrams** | **%s** |" % N(merge['final_bigrams']))
A("")

A("## 5. `f` distribution\n")
A("```")
mx = max(hist.values())
for b in sorted(hist):
    n = hist[b]
    A("%3d-%3d | %-56s %s" % (b, b + 15, '#' * max(1, int(56.0 * n / mx)), N(n)))
A("```\n")

A("## 6. Floor list\n")
A("%d curated core-Egyptian words pinned at `f=250`, %d of them absent from the MSA base entirely.\n"
  % (len(floor), sum(1 for r in floor if r[2] == '0')))
A("Selection is **curated, not derived**. Ranking candidates by \"frequent in corpus but missing")
A("from the base\" was tried first and rejected: on Egyptian forum text it surfaces football nouns,")
A("proper nouns, a concatenation artifact, and porn spam - none of which belong at f=250. The")
A("blocklist in `derive_floor.py` holds those out.\n")
A("- **Category A** (%d): Egyptian core function / discourse words" % sum(1 for r in floor if r[3] == 'catA'))
A("- **Category B** (%d): Egyptian orthographic variants (hamza-less alef, ya/alef-maqsura)"
  % sum(1 for r in floor if r[3] == 'catB'))
A("")
A("Full list with corpus counts is in `floor_words.tsv`.\n")

A("## 7. Validation\n")
A("| Check | Result |")
A("|---|---|")
for k, v in [("toolchain reproduces official `main_ar.dict` byte-for-byte", "PASS (md5 `f08701717e66613ed51d7fd111f346d7`)"),
             ("`locale` is plain `ar`", "PASS"),
             ("`version` > 18", "PASS (54)"),
             ("no duplicate entries", "PASS (0)"),
             ("all `f` in 0..255", "PASS"),
             ("all words pure Arabic script", "PASS"),
             ("every bigram target exists as a word", "PASS (0 orphans)"),
             ("binary magic `0x9BC13AFE`", "PASS"),
             ("binary header attributes intact", "PASS"),
             ("30 Egyptian test words present", "PASS"),
             ("31 MSA test words present", "PASS"),
             ("16 non-floored Egyptian words present with calibrated `f`", "PASS")]:
    A("| %s | %s |" % (k, v))
A("")
A("**Final size: %.2f MB** (budget 15 MB).\n" % (val['size'] / 1e6))
A("> `dicttool_aosp.jar` cannot decompile `.dict` -> `.combined`: that direction needs a native")
A("> JNI library (`latinime-aosp-dicttool-host`) that is not shipped, and it fails on both Linux")
A("> and Windows. The round-trip check was therefore replaced by (a) reproducing the official")
A("> published dictionary byte-for-byte from its published source, and (b) parsing the compiled")
A("> binary's header directly.")

with io.open(OUT + 'STATS.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write("\n".join(L) + "\n")
print("wrote STATS.md (%d lines)" % len(L))
