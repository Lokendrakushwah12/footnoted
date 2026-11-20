# footnoted

Question answering over documentation where every claim carries a quote from the page it came
from, and questions the docs cannot answer are declined.

The runtime built around this project is [thirdshift](https://github.com/Lokendrakushwah12/thirdshift).
Its [teardown](https://github.com/Lokendrakushwah12/thirdshift/blob/master/docs/teardown.md) gives
the verdict on every technique in both repositories, with diagrams and cost per task.

## Run it

```sh
sh scripts/fetch_corpora.sh     # Python 3.15 docs, MCP docs, Claude developer docs
uv run footnoted ingest         # chunk, index, embed (about 30 minutes on an M4 Pro)
uv run footnoted search "how do I memoize a function" --corpus python
uv run footnoted eval evals/python.jsonl evals/docs.jsonl -v
uv run python tests/test_core.py

ollama pull hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M   # free local model, 2.1 GB
uv run footnoted ask "How do I compute a SHA-256 digest?" --corpus python
uv run footnoted gate record               # run the answer evals, write evals/scorecard.json
uv run footnoted gate check                # compare with evals/golden.json; this is what CI runs
uv run footnoted chat --corpus python      # follow-up questions; remembers across sessions
uv run footnoted eval-memory evals/memory.jsonl -v
uv run footnoted graph build               # knowledge graph of the MCP specification, ~15 minutes
uv run footnoted graph query "How does a server get the user's working directories?"
uv run footnoted eval-graph evals/graph.jsonl
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
| Answerable, answered citing an expected page | 31 of 72 |
| Answerable, answered citing only other pages | 23 of 72 |
| Answerable, declined | 18 of 72 |
| Unanswerable, declined (correct) | 11 of 12 |
| Unanswerable, answered anyway | 1 of 12 |

Median 6 seconds per question; 51 questions needed at least one rewritten query.

- **Declining works; answering is limited by the model.** Retrieval puts an expected page in the
  top five for roughly 80% of questions, but the 3B model turns that into a grounded answer for 43%.
- **Matching quotes on words instead of characters** moved expected-page answers from 23 to 32. The
  model copies the words but rewrites quote marks and markup.
- **Answers citing other pages are only partly judged.** Some are right answers from a page the
  eval does not list; some are wrong. The gate below adds a key-term check for the Python
  questions, and shows that the model judge is not yet reliable enough to settle the rest.

## Context budget

`context.py` decides what goes into each prompt. Required items (the instructions, the question)
always go in. Optional items carry a slot (retrieval, memory, tools) and a priority; each slot
spends up to its share of the remaining budget, and whatever a slot leaves unused goes to the
best remaining items of any slot. The prompt is built from what was kept, so it cannot exceed
the model's window.

This matters with Ollama in particular: a prompt longer than the window is not an error there.
The start of it, system prompt included, is silently cut off.

Tokens are estimated at 3.25 characters each, the 5th percentile measured on 108 passages with
the Qwen tokenizer (median 4.25). The eval prints the largest real prompt as the check.

Same 84 questions, two policies:

| | at most 5 passages | as many as fit (up to 12) |
|---|---|---|
| Answerable, answered citing an expected page | 32 of 72 | 28 of 72 |
| Answerable, declined | 18 of 72 | 22 of 72 |
| Unanswerable, declined | 11 of 12 | 12 of 12 |
| Median seconds per question | 5.7 | 16.1 |
| Largest real prompt (4,096-token window) | 1,886 tokens | 3,479 tokens |

- **Filling the window made answers worse and nearly three times slower.** For a 3B model the
  budget is a ceiling, not a target, so the default is five passages.
- **The first estimate was too optimistic.** At 3.5 characters per token one code-heavy prompt
  came out 83 tokens over its budget; only the safety margin kept it inside the window. The
  estimate now uses the 5th percentile.
- Memory and tool slots are implemented and tested but have nothing to hold yet.

## Memory

`footnoted chat` keeps three tiers of memory (`memory.py`):

- **Working**: the last four turns, verbatim, for the current session.
- **Episodes**: when turns leave working memory, or the session ends, the model compresses them
  into a short summary.
- **Facts**: durable statements about the user ("The user stores application data in SQLite"),
  extracted in the same call.

Episodes and facts are stored in SQLite with an embedding, an importance of 1 to 5 and an expiry
date: 7 days per importance point for episodes, 30 for facts. Expired rows are deleted when a
session starts, and a memory repeated in a later session refreshes the stored copy instead of
adding a duplicate. Delete `data/memory.db` to forget everything.

Before each question is answered, the three most relevant stored memories and the working turns
are budgeted by the context assembler (60% turns, 40% memories) and used to rewrite the question
so it stands alone. A rewrite that drops what the user actually typed is discarded.

Is an expected page in the top five for the last turn, searched as typed versus rewritten with
memory? 30 cases:

| kind | cases | found as typed | found with memory |
|---|---|---|---|
| Follow-up that depends on the previous turn | 14 | 8 | 11 |
| Fact stated in an earlier session | 6 | 5 | 5 |
| Control: complete question after an unrelated turn | 10 | 10 | 10 |

- **Working memory is where the gain is.** "How do I make it ignore case?" becomes "How do I make
  the search ignore case when using a regular expression?".
- **Cross-session memory works mechanically but shows no gain here.** Memories were stored,
  survived a restart and were recalled, but five of the six questions were already findable as
  typed, so this set cannot show a benefit.
- **The controls caught a real failure.** In the first run the model twice returned the previous
  question instead of the new one. The discard rule above fixed that (8 to 10 of 10).
- **Limits of the 3B model**: it files almost everything as an episode and extracts few facts, it
  rates everything importance 3, and it left three follow-ups unresolved.
- **Similarity alone cannot pick memories.** With bge-small, matching question/memory pairs score
  0.50 to 0.72 and unrelated pairs up to 0.65, so recall is top-3 with a low floor.

## Eval gate

`gate.py` grades every step of the answer loop, not just the final answer, and blocks regressions.

Each run of the loop records its trajectory: `retrieve`, `draft`, `ground`, `verify`, then
`answer`, `decline`, or `rewrite` and around again. The gate checks three things:

1. **Legal path.** Every trajectory must be a path through the loop's graph of allowed moves,
   starting at `retrieve` and ending at `answer` or `decline`. Answering without `verify` fails.
2. **Step-level scores against a golden run.** `evals/golden.json` holds the accepted run.
   The current run may fall at most two questions below it on "cites an expected page" and
   "correct by key term", and one on "declines the unanswerable".
3. **Freshness.** The scorecard stores a hash of the answer-loop sources and eval files. If
   either changed since it was recorded, the gate fails until the evals are re-run.

The model runs locally, so `gate record` is run by hand and the scorecard is committed. CI
(`.github/workflows/gate.yml`) runs the unit checks and `gate check`, which needs no model.
`gate accept` promotes the current scorecard to golden, as a visible change in the diff.

The golden run, step by step:

| step | result |
|---|---|
| Retrieve: an expected page reached the model | 60 of 72 |
| Answered at all | 54 of 72 |
| Answered citing an expected page | 31 of 72 |
| Correct by key term (Python questions that were answered) | 29 of 34 |
| Unanswerable questions declined | 11 of 12 |
| Claims: drafted, quote verified, kept after the check | 171, 135, 80 |
| Trajectories off the legal graph | 0 |

- **Is the band right?** Re-recording the same code moved "cites an expected page" by one
  question and passed. Cutting retrieval to one passage moved it by 12 and "correct" by 14, and
  the gate failed, listing each question that got worse.
- **Correctness is a proxy.** A Python answer counts as correct when it names an API a right
  answer must name (`TemporaryDirectory`, `sha256`). It cannot tell a right use from a wrong one.
- **The model judge is calibrated, and fails calibration.** Asked whether each answer is correct,
  the 3B model agreed with the key-term check on 26 of 34 answers and rejected 7 that named the
  right API. It is reported but never gated on. It accepts 8 of the 20 answers that have no key
  terms, which at that agreement rate says little.
- **CI trusts the committed scorecard.** It proves the scorecard is fresh, not that it was
  produced honestly. Replaying recorded model calls in CI, or a self-hosted runner with the
  model, would close that gap.

## Knowledge graph

`graph.py` turns the 29 pages of the MCP specification into a graph and uses it to find pages.

1. **Extract.** The model reads the pages five chunks at a time and returns up to six entities
   and six relations per batch. Replies are saved as they arrive, so a build can resume.
2. **Merge.** Names are normalised (case, punctuation, simple plurals) and merged into one graph;
   a relation is kept only if both ends were extracted as entities.
3. **Cluster and summarise.** Label propagation groups the graph into communities, and the model
   writes a title and summary for each.
4. **Query.** Entities named in the question, or similar to it, are the seeds. A two-hop walk
   scores their neighbourhood, damped at hubs, and pages are ranked by the entities they mention.

The graph: 394 entities, 496 relations, 19 communities, from 102 batches of which 8 failed.

Page retrieval, top five pages. A multi-hop question needs two specific pages and counts only if
both are found; a control is an ordinary single-page question.

| kind | questions | hybrid | hybrid, limited to the graph's pages | graph alone | hybrid + graph |
|---|---|---|---|---|---|
| multi-hop | 12 | 7 | 11 | 6 | 7 |
| control | 14 | 7 | 10 | 9 | 11 |

- **The graph adds nothing that narrowing the search does not.** Fusing it with hybrid search
  lifts the controls from 7 to 11, which looks like a win until hybrid search is simply limited
  to the same 29 pages: that scores 10 on the controls and 11 of 12 on multi-hop, where the graph
  scores 7. The graph was working as a scope filter.
- **The mechanism does work on individual questions.** For "How does a server get the user's
  working directories while it is handling a request?" hybrid search finds neither page; the graph
  walks from `workspace` to `roots` to `InputRequiredResult` and returns both.
- **The extraction is the weak part.** The 3B model labels 322 of 394 entities as plain
  "concept", leaves 119 with no relations, and makes `client` and `server` hubs with over 100
  edges each, which pulls 134 entities into one community.
- **Asking for too much lost data.** At eight entities and eight relations per batch, 31 of 102
  replies were cut off mid-JSON by the output cap. At six and six, 8 still are.
- **Community summaries are generated but not evaluated**, and in an earlier build two of them
  invented an expansion of "MCP". Treat them as unverified.
- The multi-hop questions were written from the pages' section headings, not from the graph.

## Limits of this eval

- 72 questions, written by the same person who built the pipeline, with no held-out split. The
  fusion weight and the per-page cap were chosen on these same questions.
- The questions deliberately avoid the page's own keywords. That is the hard case for BM25 and for
  rerankers that lean on word overlap. Keyword-style queries are not measured.
- Expected pages are incomplete: some retrieved pages are relevant but not listed.
