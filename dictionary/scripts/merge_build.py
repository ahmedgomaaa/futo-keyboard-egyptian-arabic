# -*- coding: utf-8 -*-
"""Merge the MSA base wordlist with Egyptian corpus frequencies into one .combined.

Usage:
  merge_build.py --base B.combined --counts C.tsv --bigrams B.tsv --out O.combined
                 [--min-count 5] [--eg-min-count-new 12] [--max-bigrams 4]
                 [--floor-file F.tsv] [--floor-f 250] [--report R.json]
"""
import sys, os, json, math, time, argparse, collections, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egtext

BASE_DROPPED = []
BASE_NORMALIZED = []



def normalize_entry(w):
    """Clean a base-wordlist entry: NFKC (folds Arabic presentation forms to
    normal letters), then drop tatweel/diacritics. Returns None if what is left
    is not a pure Arabic-letter token. Deliberately does NOT fold alef-hamza,
    ta-marbuta, or ya/alef-maqsura."""
    w = unicodedata.normalize("NFKC", w)
    w = egtext.RE_DIACRITIC.sub("", w)
    if not w or not egtext.RE_TOKEN.fullmatch(w):
        return None
    if len(w) > egtext.MAX_LEN:
        return None
    return w


def parse_base(path):
    words, bigrams, order = {}, collections.defaultdict(list), []
    header = None
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cur = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(" word="):
                fields = dict(kv.split("=", 1) for kv in line.strip().split(",") if "=" in kv)
                raw = fields["word"]
                w = normalize_entry(raw)
                if w is None:
                    BASE_DROPPED.append(raw)
                    cur = None
                    continue
                if w != raw:
                    BASE_NORMALIZED.append((raw, w))
                cur = w
                if w not in words:
                    order.append(w)
                words[w] = max(words.get(w, 0), int(fields.get("f", 0)))
            elif line.strip().startswith("bigram=") and cur is not None:
                fields = dict(kv.split("=", 1) for kv in line.strip().split(",") if "=" in kv)
                b = normalize_entry(fields["bigram"])
                if b is not None:
                    bigrams[cur].append((b, int(fields.get("f", 1))))
    return header, words, bigrams, order


def load_counts(path, min_count):
    c = {}
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            n = int(parts[1])
            if n < min_count:
                dropped += 1
                continue
            c[parts[0]] = n
    return c, dropped


def fit_scale(counts, base_words):
    """Least-squares fit of base_f ~ a*log(count)+b over the shared vocabulary."""
    xs, ys = [], []
    for w, n in counts.items():
        bf = base_words.get(w)
        if bf is not None and bf > 0 and n >= 5:
            xs.append(math.log(n)); ys.append(float(bf))
    n = len(xs)
    if n < 100:
        return 12.0, 0.0, 0, 0.0
    mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sxx = sum((x-mx)**2 for x in xs)
    a = sxy/sxx if sxx else 12.0
    b = my - a*mx
    syy = sum((y-my)**2 for y in ys)
    r2 = (sxy*sxy)/(sxx*syy) if sxx and syy else 0.0
    return a, b, n, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--bigrams")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--eg-min-count-new", type=int, default=5,
                    help="higher bar for words NOT in the MSA base (size control)")
    ap.add_argument("--max-bigrams", type=int, default=4)
    ap.add_argument("--bigram-parent-min-f", type=int, default=60)
    ap.add_argument("--floor-file")
    ap.add_argument("--floor-f", type=int, default=250)
    ap.add_argument("--offensive-file")
    ap.add_argument("--report")
    a = ap.parse_args()

    header, base_words, base_bigrams, base_order = parse_base(a.base)
    counts, below_min = load_counts(a.counts, a.min_count)
    slope, inter, nfit, r2 = fit_scale(counts, base_words)

    def to_f(n):
        v = int(round(slope*math.log(n) + inter))
        return max(1, min(254, v))

    merged = dict(base_words)
    stats = collections.Counter()
    for w, n in counts.items():
        ef = to_f(n)
        if w in base_words:
            if ef > base_words[w]:
                merged[w] = ef; stats["raised"] += 1
            else:
                stats["kept_base"] += 1
        else:
            if n < a.eg_min_count_new:
                stats["new_below_bar"] += 1
                continue
            merged[w] = ef; stats["new_words"] += 1

    floored = []
    if a.floor_file and os.path.exists(a.floor_file):
        with open(a.floor_file, encoding="utf-8") as f:
            for line in f:
                w = line.split("\t")[0].strip()
                if w:
                    merged[w] = max(merged.get(w, 0), a.floor_f)
                    floored.append(w)

    offensive = set()
    if a.offensive_file and os.path.exists(a.offensive_file):
        with open(a.offensive_file, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    offensive.add(w)
    offensive_hit = sorted(w for w in offensive if w in merged)

    # ---- bigrams ----
    eg_big = collections.defaultdict(list)
    if a.bigrams and os.path.exists(a.bigrams):
        tmp = collections.defaultdict(list)
        with open(a.bigrams, encoding="utf-8") as f:
            for line in f:
                p, b, c = line.rstrip("\n").split("\t")
                tmp[p].append((b, int(c)))
        for p, lst in tmp.items():
            if merged.get(p, 0) < a.bigram_parent_min_f:
                continue
            lst = [(b, c) for b, c in lst if b in merged]
            lst.sort(key=lambda x: -x[1])
            if lst:
                eg_big[p] = lst[:a.max_bigrams]

    out_bigrams = {}
    for w in merged:
        # An offensive parent keeps its entry (marked possibly_offensive) but gets
        # no next-word list: possibly_offensive does not suppress a word's bigram
        # suggestions, so leaving them in would surface more offensive terms.
        if w in offensive:
            continue
        if w in eg_big:
            cand = [b for b, _ in eg_big[w] if b not in offensive]
        elif w in base_bigrams:
            cand = [b for b, _ in sorted(base_bigrams[w], key=lambda x: x[1]) if b not in offensive]
        else:
            continue
        if cand:
            out_bigrams[w] = cand[:a.max_bigrams]

    # ---- write ----
    ts = int(time.time())
    hdr = ("dictionary=main:ar,locale=ar,description=Arabic (Egyptian),"
           "date=%d,version=54" % ts)
    nb = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(hdr + "\n")
        for w in sorted(merged, key=lambda x: (-merged[x], x)):
            if w in offensive:
                f.write(" word=%s,f=%d,possibly_offensive=true\n" % (w, merged[w]))
            else:
                f.write(" word=%s,f=%d\n" % (w, merged[w]))
            for i, b in enumerate(out_bigrams.get(w, [])):
                f.write("  bigram=%s,f=%d\n" % (b, i + 1))
                nb += 1

    rep = {
        "base_entries_normalized": len(BASE_NORMALIZED),
        "base_entries_dropped": len(BASE_DROPPED),
        "base_dropped_examples": BASE_DROPPED[:10],
        "base_normalized_examples": ["%s->%s" % (a, b) for a, b in BASE_NORMALIZED[:10]],
        "base_words": len(base_words), "base_bigrams": sum(len(v) for v in base_bigrams.values()),
        "eg_types_after_min_count": len(counts), "eg_types_below_min_count": below_min,
        "fit_slope": round(slope, 4), "fit_intercept": round(inter, 4),
        "fit_n_shared": nfit, "fit_r2": round(r2, 4),
        "raised": stats["raised"], "kept_base": stats["kept_base"],
        "new_words": stats["new_words"], "new_below_bar": stats["new_below_bar"],
        "floored": len(floored),
        "marked_offensive": len(offensive_hit),
        "offensive_examples": offensive_hit[:8],
        "final_words": len(merged), "final_bigrams": nb,
        "header": hdr,
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
