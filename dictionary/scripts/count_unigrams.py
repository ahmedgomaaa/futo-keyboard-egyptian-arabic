# -*- coding: utf-8 -*-
"""Count Arabic token frequencies for one corpus, in parallel.

Usage: count_unigrams.py <corpus_name> <out_dir> <path> [path ...]
Writes <out_dir>/counts_<corpus>.tsv  (token<TAB>count, sorted desc)
       <out_dir>/stats_<corpus>.json
"""
import sys, os, json, collections, multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egtext

NSHARD_BIG = 24          # shards for a single large file
DROP_EXAMPLE_CAP = 400   # distinct dropped tokens to remember per reason


def _iter_records(path, shard, nshards):
    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        col = "text" if "text" in cols else cols[0]
        for rg in range(pf.num_row_groups):
            if rg % nshards != shard:
                continue
            tbl = pf.read_row_group(rg, columns=[col])
            for v in tbl.column(0).to_pylist():
                if v:
                    yield v
    else:
        reader = egtext.read_gz if low.endswith(".gz") else egtext.read_txt
        for i, line in enumerate(reader(path)):
            if nshards == 1 or i % nshards == shard:
                yield line


def worker(task):
    path, shard, nshards = task
    counts = collections.Counter()
    stats = collections.Counter()
    drops = {"too_long": collections.Counter(), "elongation": collections.Counter()}
    for rec in _iter_records(path, shard, nshards):
        for t in egtext.tokenize(rec):
            stats["raw_tokens"] += 1
            reason = egtext.token_ok(t)
            if reason:
                stats["dropped_" + reason] += 1
                d = drops[reason]
                if len(d) < DROP_EXAMPLE_CAP:
                    d[t] += 1
                continue
            counts[t] += 1
    return counts, stats, drops


def main():
    corpus, outdir = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)

    tasks = []
    if len(paths) == 1:
        tasks = [(paths[0], s, NSHARD_BIG) for s in range(NSHARD_BIG)]
    else:
        tasks = [(p, 0, 1) for p in paths]

    total = collections.Counter()
    stats = collections.Counter()
    drops = {"too_long": collections.Counter(), "elongation": collections.Counter()}

    nproc = min(24, len(tasks))
    with mp.Pool(nproc) as pool:
        done = 0
        for c, s, d in pool.imap_unordered(worker, tasks):
            total.update(c); stats.update(s)
            for k in drops:
                drops[k].update(d[k])
            done += 1
            sys.stderr.write("\r  %s: %d/%d shards, %d unique" % (corpus, done, len(tasks), len(total)))
            sys.stderr.flush()
    sys.stderr.write("\n")

    with open(os.path.join(outdir, "counts_%s.tsv" % corpus), "w", encoding="utf-8", newline="\n") as f:
        for t, c in total.most_common():
            f.write("%s\t%d\n" % (t, c))

    out = {
        "corpus": corpus,
        "files": len(paths),
        "raw_tokens": stats["raw_tokens"],
        "dropped_too_long": stats["dropped_too_long"],
        "dropped_elongation": stats["dropped_elongation"],
        "kept_tokens": sum(total.values()),
        "unique_kept": len(total),
        "examples_too_long": [w for w, _ in drops["too_long"].most_common(15)],
        "examples_elongation": [w for w, _ in drops["elongation"].most_common(15)],
    }
    with open(os.path.join(outdir, "stats_%s.json" % corpus), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("examples")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
