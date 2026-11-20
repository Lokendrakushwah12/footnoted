# footnoted

Question answering over documentation where every claim carries a quote from the page it came
from, and questions the docs cannot answer are declined.

## Run it

```sh
sh scripts/fetch_corpora.sh     # Python 3.15 docs, MCP docs, Claude developer docs
uv run footnoted ingest         # chunk, index, embed (about 30 minutes on an M4 Pro)
uv run footnoted search "how do I memoize a function" --corpus python
uv run footnoted eval evals/python.jsonl evals/docs.jsonl -v
uv run python tests/test_core.py

ollama pull hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M   # free local model, 2.1 GB
uv run footnoted ask "How do I compute a SHA-256 digest?" --corpus python
uv run footnoted eval-ask evals/python.jsonl evals/docs.jsonl evals/unanswerable.jsonl
```

Any OpenAI-compatible endpoint works: set `FOOTNOTED_BASE_URL`, `FOOTNOTED_MODEL` and
`FOOTNOTED_API_KEY`.

## How it works

- **Chunks**: pages are split on headings, then packed into pieces of at most 1,200 characters.
  Each chunk is indexed with its page title and section heading in front.
- **BM25**: SQLite FTS5, with the title and heading weighted double.
- **Dense**: `bge-small-en-v1.5` vectors in one numpy matrix, searched by brute force.
- **Hybrid**: reciprocal rank fusion of the two, dense weighted 2x.
- **Rerank**: optional cross-encoder (`ms-marco-MiniLM-L-6-v2`) over the top 20 fused results.
- **Filters**: `--corpus` restricts a search to one doc set; at most two chunks per page are returned.

The index is 31,777 chunks from 1,040 pages.

## Results

A question counts as answered when a chunk from an expected page is in the top results.

Python docs, 46 questions:

| mode | recall@1 | recall@5 | MRR@10 | p50 ms |
|---|---|---|---|---|
| bm25 | 0.52 | 0.78 | 0.64 | 14 |
| dense | 0.59 | 0.85 | 0.70 | 10 |
| hybrid | 0.63 | 0.87 | 0.74 | 20 |
| rerank | 0.50 | 0.85 | 0.64 | 548 |

MCP and Claude docs, 26 questions:

| mode | recall@1 | recall@5 | MRR@10 | p50 ms |
|---|---|---|---|---|
| bm25 | 0.35 | 0.65 | 0.48 | 7 |
| dense | 0.42 | 0.73 | 0.56 | 4 |
| hybrid | 0.35 | 0.58 | 0.48 | 13 |
| rerank | 0.38 | 0.65 | 0.48 | 310 |

## What the numbers say

- **No mode wins on both sets.** Hybrid is best on Python (two more questions right at rank 1 than
  dense). Dense is best on the MCP and Claude set, where fusing in BM25 costs four questions at
  recall@5. At this sample size neither gap is conclusive.
- **Reranking did not pay off.** Three cross-encoders were tried on the same candidates; none beat
  the best first-stage mode on recall@5. `bge-reranker-base` raised recall@1 on the MCP and Claude
  set to 0.50, at 2.1 seconds per query.
- **Dropping question words from the keyword query mattered most for BM25**: recall@5 on Python went
  from 0.61 to 0.78, because "how do I" was matching FAQ headings phrased the same way.
- **Capping chunks per page helped** on both sets; long pages were filling the top five.

## Answering

`footnoted ask` runs a loop of at most three steps:

1. **Retrieve** five passages for the current query.
2. **Draft** an answer as a list of claims; each claim names a passage and quotes it.
3. **Ground**: a claim is kept only if its quote's words appear, in order, in a retrieved passage.
4. **Verify**: a second model call checks that the claims answer the question and that each
   statement is backed by its quote.
5. If nothing survives, **rewrite** the query and try again; after the last step, **decline**.

Results with the local 3B model, 84 questions:

| | count |
|---|---|
| Answerable, answered citing an expected page | 32 of 72 |
| Answerable, answered citing only other pages | 22 of 72 |
| Answerable, declined | 18 of 72 |
| Unanswerable, declined (correct) | 11 of 12 |
| Unanswerable, answered anyway | 1 of 12 |

Median 7 seconds per question; 51 questions needed at least one rewritten query.

- **Declining works; answering is limited by the model.** Retrieval puts an expected page in the
  top five for roughly 80% of questions, but the 3B model turns that into a grounded answer for 44%.
- **Matching quotes on words instead of characters** moved expected-page answers from 23 to 32. The
  model copies the words but rewrites quote marks and markup.
- **The 22 answers citing other pages are not judged.** Some are right answers from a page the eval
  does not list; some are wrong. Telling them apart needs a judge, which is not built yet.

## Limits of this eval

- 72 questions, written by the same person who built the pipeline, with no held-out split. The
  fusion weight and the per-page cap were chosen on these same questions.
- The questions deliberately avoid the page's own keywords. That is the hard case for BM25 and for
  rerankers that lean on word overlap. Keyword-style queries are not measured.
- Expected pages are incomplete: some retrieved pages are relevant but not listed.
