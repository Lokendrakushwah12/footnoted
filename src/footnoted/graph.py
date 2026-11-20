"""GraphRAG over one slice of the docs.

build   the model reads the pages in batches and returns entities and relations; names are
        normalised and merged into one graph; label propagation groups it into communities;
        the model writes a summary of each.
query   entities named in the question, or similar to it, are the seeds; a two-hop walk scores
        their neighbourhood; pages are ranked by the entities they mention.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .index import rrf
from .llm import ask

GRAPH, RAW = Path("data/graph.json"), Path("data/graph_raw.jsonl")
# ponytail: the MCP specification only (29 pages, about 95 model calls). The schema reference and
# changelog are left out as near-duplicate noise. Widen the filter to graph more of the corpus.
SCOPE = "corpus = 'mcp' and doc like 'specification/%' and doc not like '%/schema' and doc not like '%/changelog'"
BATCH = 5  # chunks per extraction call, about 1,500 tokens
TYPES = ["role", "feature", "method", "message", "capability", "transport", "error", "concept"]

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": TYPES},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "relation", "target"],
            },
        },
    },
    "required": ["entities", "relations"],
}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
}

# Six and six, with short descriptions, because the reply must fit the 600-token output cap:
# asking for eight of each left 31 of 102 batches cut off mid-JSON and lost.
# The example is about chess on purpose: nothing in it can be mistaken for the documentation,
# and an entity called "castling" in the graph would show the example leaking.
EXTRACT_PROMPT = """You extract a knowledge graph from protocol documentation. Return at most 6 \
entities and at most 6 relations, using only what the text says.
- entities: the named things the text defines or relies on: roles, features, methods, messages, \
capabilities, transports, errors, concepts. name is the short name exactly as the text writes it. \
description is one sentence of at most 12 words.
- relations: how two of your entities are connected. source and target must be names from your \
entity list. relation is a short verb phrase such as "sends", "requires", "is part of".

Example for a text about chess:
{"entities": [{"name": "castling", "type": "concept", "description": "A move of the king and a \
rook together."}, {"name": "king", "type": "role", "description": "The piece that must not be \
captured."}], "relations": [{"source": "castling", "relation": "moves", "target": "king"}]}"""

SUMMARY_PROMPT = """These entities and relations from protocol documentation form one cluster. \
Give the cluster a short title, and summarise in two or three sentences what it covers and how \
its parts fit together. Use only what is listed."""


def norm(name: str) -> str:
    """Canonical key for a name: lower case, punctuation dropped, simple plurals folded."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def merge(replies: dict) -> dict:
    """Turn per-batch extractions, keyed by (page, batch), into one graph."""
    entities: dict[str, dict] = {}
    edges = set()
    for (doc, _), reply in sorted(replies.items()):
        for e in reply.get("entities", []):
            key = norm(e["name"])
            if len(key) < 2:
                continue
            kept = entities.setdefault(key, {"name": e["name"], "type": e["type"], "description": "", "docs": []})
            if len(e["description"]) > len(kept["description"]):
                kept["description"] = e["description"]
            if doc not in kept["docs"]:
                kept["docs"].append(doc)
        for r in reply.get("relations", []):
            edges.add((norm(r["source"]), r["relation"].strip().lower(), norm(r["target"])))
    # A relation is kept only if both ends were extracted as entities somewhere.
    # ponytail: names merge on spelling alone. "MRTR" and "multi round-trip request" stay two
    # nodes; merge on embedding similarity if split entities turn out to hurt.
    kept_edges = sorted(e for e in edges if e[0] in entities and e[2] in entities and e[0] != e[2])
    return {"entities": entities, "edges": kept_edges}


def communities(nodes: list[str], edges: list, rounds: int = 20) -> list[list[str]]:
    """Label propagation: every node repeatedly takes the label most common among its neighbours."""
    # ponytail: flat clusters and no modularity guarantee. Leiden gives the hierarchy the
    # original GraphRAG uses; this is twenty lines and has no dependency.
    neighbours = defaultdict(set)
    for source, _, target in edges:
        neighbours[source].add(target)
        neighbours[target].add(source)
    label = {n: n for n in nodes}
    rng = random.Random(0)
    for _ in range(rounds):
        order = sorted(nodes)
        rng.shuffle(order)
        changed = False
        for node in order:
            counts = Counter(label[m] for m in neighbours[node])
            if counts:
                best = max(sorted(counts), key=counts.__getitem__)
                changed |= best != label[node]
                label[node] = best
        if not changed:
            break
    groups = defaultdict(list)
    for node in sorted(nodes):
        groups[label[node]].append(node)
    return sorted((g for g in groups.values() if len(g) >= 3), key=len, reverse=True)


def build(index) -> dict:
    pages = defaultdict(list)
    for row in index.db.execute(f"select doc, title, heading, text from chunks where {SCOPE} order by id"):
        pages[row["doc"]].append(row)
    # Every reply is appended to RAW as it arrives, so an interrupted build resumes where it stopped.
    replies = {}
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            saved = json.loads(line)
            replies[(saved["doc"], saved["batch"])] = saved["reply"]
    with RAW.open("a") as raw:
        for doc, rows in pages.items():
            for start in range(0, len(rows), BATCH):
                key = (doc, start // BATCH)
                if replies.get(key):  # an empty reply was cut off mid-JSON: try it again
                    continue
                text = "\n\n".join(f"## {r['heading']}\n{r['text']}" for r in rows[start : start + BATCH])
                replies[key] = ask(EXTRACT_PROMPT, f"Page: {rows[0]['title']}\n\n{text}", EXTRACT_SCHEMA)
                raw.write(json.dumps({"doc": doc, "batch": key[1], "reply": replies[key]}) + "\n")
                raw.flush()
                print(".", end="", flush=True)
    graph = merge(replies)
    graph["failed_batches"] = sum(not r for r in replies.values())
    graph["communities"] = []
    for members in communities(list(graph["entities"]), graph["edges"]):
        inside = set(members)
        listing = "\n".join(f"- {graph['entities'][m]['name']}: {graph['entities'][m]['description']}" for m in members[:25])
        links = "\n".join(f"- {s} {r} {t}" for s, r, t in graph["edges"] if s in inside and t in inside)
        reply = ask(SUMMARY_PROMPT, f"Entities:\n{listing}\n\nRelations:\n{links[:2500]}", SUMMARY_SCHEMA)
        graph["communities"].append({"members": members, "title": reply.get("title", ""), "summary": reply.get("summary", "")})
        print("+", end="", flush=True)
    print()
    GRAPH.write_text(json.dumps(graph))
    return graph


class Graph:
    def __init__(self, data: dict, embed: Callable[[list[str]], np.ndarray] | None = None) -> None:
        self.entities, self.communities = data["entities"], data.get("communities", [])
        self.keys = list(self.entities)
        self.neighbours: dict[str, set[str]] = defaultdict(set)
        for source, _, target in data["edges"]:
            self.neighbours[source].add(target)
            self.neighbours[target].add(source)
        self.embed, self.vectors = embed, None
        if embed and self.keys:
            # ponytail: embedded on every load, a few seconds for a few hundred entities.
            # Save the matrix next to the graph if the graph grows.
            self.vectors = self._unit(embed([f"{e['name']}: {e['description']}" for e in self.entities.values()]))

    @staticmethod
    def _unit(vectors) -> np.ndarray:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def seeds(self, question: str, similar: int = 3) -> dict[str, float]:
        """Starting entities: those named in the question, plus the few most similar to it."""
        words = f" {norm(question)} "
        # Rarer entities are stronger seeds: one on every page says little about any of them.
        found = {
            key: 1 / len(self.entities[key]["docs"]) ** 0.5
            for key in self.keys
            if len(key) > 3 and f" {key} " in words
        }
        if self.vectors is not None:
            scores = self.vectors @ self._unit(self.embed([question]))[0]
            for i in np.argsort(-scores)[:similar]:
                found.setdefault(self.keys[i], float(scores[i]) / len(self.entities[self.keys[i]]["docs"]) ** 0.5)
        return found

    def walk(self, seeds: dict[str, float], hops: int = 2, decay: float = 0.5) -> dict[str, float]:
        """Spread seed scores outward. Each hop halves the score and divides it by the square root
        of the node's degree, so a hub like "server" does not light up the whole graph."""
        score, frontier = dict(seeds), dict(seeds)
        for _ in range(hops):
            reached: dict[str, float] = {}
            for node, value in frontier.items():
                passed = value * decay / max(1, len(self.neighbours[node])) ** 0.5
                for other in self.neighbours[node]:
                    if passed > score.get(other, 0) and passed > reached.get(other, 0):
                        reached[other] = passed
            score.update(reached)
            frontier = reached
        return score

    def pages(self, question: str) -> list[str]:
        """Pages ranked by the scored entities they mention."""
        totals: dict[str, float] = defaultdict(float)
        for key, value in self.walk(self.seeds(question)).items():
            docs = self.entities[key]["docs"]
            for doc in docs:
                totals[doc] += value / len(docs)
        return sorted(totals, key=totals.__getitem__, reverse=True)


def hybrid_pages(index, question: str, corpus: str, depth: int = 20) -> list[str]:
    """Pages in the order hybrid search first reaches them."""
    return list(dict.fromkeys(h["doc"] for h in index.search(question, corpus, "hybrid", k=depth, per_doc=depth)))


def combined_pages(index, graph: Graph, question: str, corpus: str) -> list[str]:
    return rrf(hybrid_pages(index, question, corpus), graph.pages(question)[:10])


def explain(graph: Graph, question: str) -> str:
    """What the graph makes of a question: seeds, what the walk reached, pages, and a cluster summary."""
    seeds = graph.seeds(question)
    scores = graph.walk(seeds)
    name = lambda key: graph.entities[key]["name"]
    reached = sorted((k for k in scores if k not in seeds), key=scores.__getitem__, reverse=True)[:8]
    out = [
        "seeds:   " + ", ".join(name(k) for k in sorted(seeds, key=seeds.__getitem__, reverse=True)),
        "reached: " + ", ".join(name(k) for k in reached),
        "pages:",
        *[f"  {doc}" for doc in graph.pages(question)[:5]],
    ]
    top = max(seeds, key=seeds.__getitem__, default=None)
    for community in graph.communities:
        if top in community["members"]:
            out += ["", f"cluster: {community['title']}", community["summary"]]
            break
    return "\n".join(out)


def load(index) -> Graph:
    return Graph(json.loads(GRAPH.read_text()), lambda texts: np.array(list(index.embedder.embed(texts))))
