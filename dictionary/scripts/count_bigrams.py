# -*- coding: utf-8 -*-
"""Count next-word continuations for high-frequency parent words.

Usage: count_bigrams.py <out_dir> <parent_vocab_tsv> <n_parents> <n_next> <path> [path ...]
Writes <out_dir>/bigrams.tsv  (parent<TAB>next<TAB>count)

Memory is bounded by pruning each parent's continuation counter to TOP_KEEP
entries every PRUNE_EVERY records.
"""
import sys, os, collections, multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egtext

TOP_KEEP = 60
PRUNE_EVERY = 50000
NSHARD_BIG = 12

PARENTS = None
NEXTS = None


def _init(parents, nexts):
    global PARENTS, NEXTS
    PARENTS, NEXTS = parents, nexts


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
            for v in pf.read_row_group(rg, columns=[col]).column(0).to_pylist():
                if v:
                    yield v
    else:
        reader = egtext.read_gz if low.endswith(".gz") else egtext.read_txt
        for i, line in enumerate(reader(path)):
            if nshards == 1 or i % nshards == shard:
                yield line


def _prune(table):
    for p, c in list(table.items()):
        if len(c) > TOP_KEEP:
            table[p] = collections.Counter(dict(c.most_common(TOP_KEEP)))


def worker(task):
    path, shard, nshards = task
    table = collections.defaultdict(collections.Counter)
    n = 0
    for rec in _iter_records(path, shard, nshards):
        toks = [t for t in egtext.tokenize(rec) if not egtext.token_ok(t)]
        for a, b in zip(toks, toks[1:]):
            if a in PARENTS and b in NEXTS:
                table[a][b] += 1
        n += 1
        if n % PRUNE_EVERY == 0:
            _prune(table)
    _prune(table)
    return {p: dict(c) for p, c in table.items()}


def main():
    outdir, vocab_path = sys.argv[1], sys.argv[2]
    n_parents, n_next = int(sys.argv[3]), int(sys.argv[4])
    paths = sys.argv[5:]

    words = []
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            w = line.split("\t")[0]
            words.append(w)
    parents = set(words[:n_parents])
    nexts = set(words[:n_next])
    sys.stderr.write("parents=%d nexts=%d\n" % (len(parents), len(nexts)))

    tasks = ([(paths[0], s, NSHARD_BIG) for s in range(NSHARD_BIG)]
             if len(paths) == 1 else [(p, 0, 1) for p in paths])

    merged = collections.defaultdict(collections.Counter)
    nproc = min(12, len(tasks))
    with mp.Pool(nproc, initializer=_init, initargs=(parents, nexts)) as pool:
        done = 0
        for part in pool.imap_unordered(worker, tasks):
            for p, c in part.items():
                merged[p].update(c)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                for p, c in list(merged.items()):
                    if len(c) > TOP_KEEP:
                        merged[p] = collections.Counter(dict(c.most_common(TOP_KEEP)))
            sys.stderr.write("\r  bigrams: %d/%d shards, %d parents" % (done, len(tasks), len(merged)))
            sys.stderr.flush()
    sys.stderr.write("\n")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "bigrams.tsv")
    npairs = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(merged, key=lambda x: -sum(merged[x].values())):
            for b, c in merged[p].most_common(20):
                f.write("%s\t%s\t%d\n" % (p, b, c))
                npairs += 1
    print("wrote %s: %d parents, %d pairs" % (out, len(merged), npairs))


if __name__ == "__main__":
    main()
