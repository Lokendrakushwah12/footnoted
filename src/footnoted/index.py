"""Build and query the index: SQLite FTS5 for BM25, a numpy matrix for dense vectors."""

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .corpus import chunks

DATA = Path("data")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODES = ("bm25", "dense", "hybrid", "rerank")
# Measured on evals/: weighting dense 2x over BM25 in the fusion beat equal weights on both sets.
DENSE_WEIGHT = 2
STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from how i in is it its me my of on or "
    "should so that the their them there these this to was what when where which who why will "
    "with would you your".split()
)


def ingest(corpora: Path = Path("corpora")) -> int:
    DATA.mkdir(exist_ok=True)
    (DATA / "index.db").unlink(missing_ok=True)
    db = sqlite3.connect(DATA / "index.db")
    db.executescript(
        """
        create table chunks(id integer primary key, corpus, doc, title, heading, text);
        create virtual table fts using fts5(context, text, content='', tokenize='porter unicode61');
        """
    )
    texts = []
    for i, c in enumerate(chunks(corpora), start=1):
        context = f"{c.title} > {c.heading}" if c.heading != c.title else c.title
        db.execute("insert into chunks values (?,?,?,?,?,?)", (i, *c))
        db.execute("insert into fts(rowid, context, text) values (?,?,?)", (i, context, c.text))
        texts.append(f"{context}\n{c.text}")
    db.commit()
    # Row i of the matrix is chunk id i + 1.
    emb = np.array(list(TextEmbedding(EMBED_MODEL).embed(texts, batch_size=64)), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(DATA / "emb.npy", emb)
    return len(texts)


def rrf(*rankings: list[int], weights: tuple[float, ...] | None = None, k: int = 60) -> list[int]:
    """Reciprocal rank fusion: merges rankings without needing comparable scores."""
    score: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights or [1] * len(rankings)):
        for rank, item in enumerate(ranking):
            score[item] += weight / (k + rank + 1)
    return sorted(score, key=score.__getitem__, reverse=True)


class Index:
    def __init__(self) -> None:
        self.db = sqlite3.connect(DATA / "index.db")
        self.db.row_factory = sqlite3.Row
        self.emb = np.load(DATA / "emb.npy")
        meta = self.db.execute("select corpus, doc from chunks order by id").fetchall()
        self.corpus = np.array([r["corpus"] for r in meta])
        self.doc = [r["doc"] for r in meta]
        self.embedder = TextEmbedding(EMBED_MODEL)
        self._reranker: TextCrossEncoder | None = None

    def bm25(self, query: str, corpus: str | None, n: int) -> list[int]:
        # Without this, "how do I ..." questions match FAQ headings phrased the same way.
        terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
        if not terms:
            return []
        # Quoted terms joined with OR: no FTS5 syntax can leak in from the query.
        match = " OR ".join(f'"{t}"' for t in terms)
        where = "and c.corpus = ?" if corpus else ""
        rows = self.db.execute(
            f"select fts.rowid from fts join chunks c on c.id = fts.rowid where fts match ? {where} "
            "order by bm25(fts, 2.0, 1.0) limit ?",  # context column counts double
            (match, corpus, n) if corpus else (match, n),
        )
        return [r[0] for r in rows]

    def dense(self, query: str, corpus: str | None, n: int) -> list[int]:
        scores = self.emb @ next(iter(self.embedder.query_embed(query)))
        if corpus:
            scores = np.where(self.corpus == corpus, scores, -np.inf)
        # ponytail: brute-force scan, about 1 ms at 20k chunks. Past ~500k chunks, switch to
        # an ANN index (sqlite-vec or FAISS).
        top = np.argsort(-scores)[:n]
        return [int(i) + 1 for i in top if np.isfinite(scores[i])]

    def fetch(self, ids: list[int]) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in self.db.execute(f"select * from chunks where id in ({marks})", ids)}
        return [rows[i] for i in ids]

    def rerank(self, query: str, ids: list[int]) -> list[int]:
        if self._reranker is None:
            self._reranker = TextCrossEncoder(RERANK_MODEL)
        texts = [f"{r['title']} > {r['heading']}\n{r['text']}" for r in self.fetch(ids)]
        scores = list(self._reranker.rerank(query, texts))
        return [i for _, i in sorted(zip(scores, ids), reverse=True)]

    def cap(self, ids: list[int], per_doc: int) -> list[int]:
        """Keep at most `per_doc` chunks from any one page, so one long page cannot fill the results."""
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for i in ids:
            doc = self.doc[i - 1]
            if seen[doc] < per_doc:
                seen[doc] += 1
                kept.append(i)
        return kept

    def search(
        self,
        query: str,
        corpus: str | None = None,
        mode: str = "hybrid",
        k: int = 5,
        pool: int = 50,  # candidates taken from each of BM25 and dense
        depth: int = 20,  # fused candidates the cross-encoder scores; the latency knob
        per_doc: int = 2,
    ) -> list[sqlite3.Row]:
        if mode == "bm25":
            ids = self.bm25(query, corpus, pool)
        elif mode == "dense":
            ids = self.dense(query, corpus, pool)
        else:
            bm25, dense = self.bm25(query, corpus, pool), self.dense(query, corpus, pool)
            ids = rrf(bm25, dense, weights=(1, DENSE_WEIGHT))
        ids = self.cap(ids, per_doc)
        if mode == "rerank":
            ids = self.rerank(query, ids[:depth])
        return self.fetch(ids[:k])
