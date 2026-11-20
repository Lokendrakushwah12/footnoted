"""Retrieval eval: for each question, is an expected page among the top results?"""

import json
import statistics
import time
from pathlib import Path

from .index import MODES, Index

K = 10


def run(path: Path, index: Index, across: bool = False) -> list[dict]:
    """Score every mode on one eval file. `across` searches all corpora instead of the question's own."""
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    known = {r[0] for r in index.db.execute("select distinct doc from chunks")}
    if missing := {d for c in cases for d in c["docs"]} - known:
        raise SystemExit(f"{path}: expected pages not in the index: {sorted(missing)}")

    results = []
    for mode in MODES:
        index.search("warm up", mode=mode)  # keep model loading out of the timings
        ranks, times = [], []
        for case in cases:
            start = time.perf_counter()
            hits = index.search(case["q"], None if across else case["corpus"], mode, K)
            times.append((time.perf_counter() - start) * 1000)
            # Rank of the first chunk from an expected page, or None for a miss.
            ranks.append(next((n for n, h in enumerate(hits, 1) if h["doc"] in case["docs"]), None))
        results.append(
            {
                "mode": mode,
                "recall@1": sum(r == 1 for r in ranks) / len(cases),
                "recall@5": sum(r is not None and r <= 5 for r in ranks) / len(cases),
                "mrr@10": sum(1 / r for r in ranks if r) / len(cases),
                "p50_ms": statistics.median(times),
                "misses": [c["q"] for c, r in zip(cases, ranks) if r is None or r > 5],
            }
        )
    return results


def run_memory(path: Path, index: Index, verbose: bool = False) -> str:
    """Memory eval: is an expected page in the top 5 for the last turn, searched as typed versus
    rewritten with memory?

    followup  the last turn only makes sense given the turn before it (working memory)
    session   the first turn states a fact in one session; the question comes in a new session
    control   the last turn is already complete and the earlier turn is unrelated (must not get worse)
    """
    import tempfile

    from .memory import Memory, standalone

    embed = lambda text: next(iter(index.embedder.embed([text])))
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    hit = lambda case, query: any(h["doc"] in case["docs"] for h in index.search(query, case["corpus"], k=5))
    tally: dict[str, list[tuple[bool, bool]]] = {}
    notes = []
    for case in cases:
        *earlier, final = case["turns"]
        with tempfile.TemporaryDirectory() as folder:  # every case starts with an empty memory
            store = Path(folder) / "memory.db"
            memory = Memory(embed, store)
            for turn in earlier:
                memory.add_turn(turn, "Noted." if case["kind"] == "session" else "(answered from the documentation)")
            if case["kind"] == "session":
                memory.flush()
                memory = Memory(embed, store)  # a new session: nothing in working memory
            rewritten = standalone(final, memory)
        result = (hit(case, final), hit(case, rewritten))
        tally.setdefault(case["kind"], []).append(result)
        notes.append(f"  [{case['kind']}] {'ok ' if result[1] else 'MISS'} {final}\n      -> {rewritten}")
        print(".", end="", flush=True)

    out = ["", "| kind | questions | found as typed | found with memory |", "|---|---|---|---|"]
    for kind, results in tally.items():
        out.append(f"| {kind} | {len(results)} | {sum(a for a, _ in results)} | {sum(b for _, b in results)} |")
    return "\n".join(out + ([""] + notes if verbose else []))


def report(path: Path, results: list[dict], verbose: bool = False) -> str:
    out = [f"{path} ", "", "| mode | recall@1 | recall@5 | MRR@10 | p50 ms |", "|---|---|---|---|---|"]
    for r in results:
        out.append(
            f"| {r['mode']} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | {r['mrr@10']:.2f} | {r['p50_ms']:.0f} |"
        )
    if verbose:
        for r in results:
            out += ["", f"{r['mode']} misses at 5:"] + [f"  - {q}" for q in r["misses"]]
    return "\n".join(out)
