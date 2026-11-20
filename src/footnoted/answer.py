"""Answer a question from retrieved passages. Every claim must carry a verbatim quote, or it is dropped."""

import os
import re
import sqlite3
from dataclasses import dataclass

from .context import Item, assemble
from .index import Index
from .llm import MAX_TOKENS, ask

MAX_STEPS = 3  # one retrieval with the question, then up to two with rewritten queries
# Ollama's default window. A longer prompt is not an error there: the start of it, system
# prompt included, is silently cut off. So the budget is enforced here.
CONTEXT_WINDOW = int(os.environ.get("FOOTNOTED_CONTEXT", "4096"))
PROMPT_BUDGET = CONTEXT_WINDOW - MAX_TOKENS - 100  # leave room for the reply and the chat template
CANDIDATES = 12  # passages retrieved per step; the assembler keeps as many as fit
MIN_QUOTE = 15  # characters; stops a one-word "quote" from counting as evidence

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "passage": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["statement", "passage", "quote"],
            },
        },
    },
    "required": ["answerable", "claims"],
}
REWRITE_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answers_question": {"type": "boolean"},
        "unsupported": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answers_question", "unsupported"],
}

ANSWER_PROMPT = """You answer questions using only the numbered passages you are given.
- Use only facts stated in the passages. Never use what you already know.
- Break the answer into short claims. Each claim has a statement (a full sentence that helps \
answer the question), the number of the passage it comes from, and a quote copied character for \
character from that passage that supports the statement.
- If the passages do not contain the answer, set answerable to false and return no claims. \
Passages on a related topic that do not answer the question do not count.

Example. Question: "How long before a request times out?" Passage [2] says "The default timeout \
is 30 seconds and can be changed per request." Reply:
{"answerable": true, "claims": [{"statement": "Requests time out after 30 seconds by default.", \
"passage": 2, "quote": "The default timeout is 30 seconds"}]}"""

VERIFY_PROMPT = """You check a drafted answer made of numbered statements, each with its evidence.
- answers_question: true only if the statements directly answer the question that was asked. \
False if they are about a different tool, library or topic, or only loosely related.
- unsupported: the numbers of statements whose evidence does not say what the statement claims. \
Empty if every statement is backed by its evidence."""

REWRITE_PROMPT = """A documentation search did not find passages that answer the question.
Write one new search query that is more likely to find them. Use the technical terms, function \
names or module names the documentation itself would use. Do not repeat an earlier query."""


@dataclass
class Claim:
    statement: str
    quote: str
    source: sqlite3.Row


@dataclass
class Answer:
    question: str
    claims: list[Claim]  # empty means the system declined to answer
    queries: list[str]  # the question, then each rewritten query
    dropped: int  # claims discarded: quote not found in a passage, or rejected by the check

    def render(self) -> str:
        if not self.claims:
            return "I can't answer that from the indexed documentation."
        pages = list(dict.fromkeys((c.source["doc"], c.source["heading"]) for c in self.claims))
        body = " ".join(f"{c.statement} [{pages.index((c.source['doc'], c.source['heading'])) + 1}]" for c in self.claims)
        notes = "\n".join(f"[{n}] {doc} | {heading}" for n, (doc, heading) in enumerate(pages, 1))
        return f"{body}\n\n{notes}"


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def grounded(quote: str, passage: str) -> bool:
    """True when the quote's words appear in the passage, in order. Punctuation and case are ignored,
    because models rewrite quote marks and markup even when they copy the words faithfully."""
    quote = _words(quote)
    return len(quote) >= MIN_QUOTE and f" {quote} " in f" {_words(passage)} "


def verify(question: str, claims: list[Claim]) -> list[Claim]:
    """Second opinion on a draft. A real quote proves the evidence exists, not that it supports the
    statement or that the statement answers this question. Returns the claims that survive."""
    drafted = "\n".join(f"{n}. {c.statement} (evidence: '{c.quote}')" for n, c in enumerate(claims, 1))
    verdict = ask(VERIFY_PROMPT, f"Question: {question}\n\nStatements:\n{drafted}", VERIFY_SCHEMA)
    if not verdict.get("answers_question"):
        return []
    return [c for n, c in enumerate(claims, 1) if n not in verdict.get("unsupported", [])]


def _passage(hit: sqlite3.Row) -> str:
    # Double quotes become single: a small model ends its JSON string at the first one it copies.
    return f"{hit['title']} > {hit['heading']}\n{hit['text']}".replace('"', "'")


def answer(index: Index, question: str, corpus: str | None = None) -> Answer:
    queries, dropped = [question], 0
    for step in range(MAX_STEPS):
        head = f"Question: {question}\n\nPassages:"
        found = index.search(queries[-1], corpus, k=CANDIDATES)
        context = assemble(
            [Item("system", ANSWER_PROMPT, required=True), Item("question", head, required=True)]
            + [Item("retrieval", _passage(h), priority=1 / rank, ref=h) for rank, h in enumerate(found, 1)],
            PROMPT_BUDGET,
        )
        hits = [item.ref for item in context.of("retrieval")]
        passages = "\n\n".join(f"[{n}] {_passage(h)}" for n, h in enumerate(hits, 1))
        draft = ask(ANSWER_PROMPT, f"{head}\n{passages}", ANSWER_SCHEMA)
        claims = []
        for c in draft.get("claims", []) if draft.get("answerable") else []:
            # Try the passage the model cited first, then the others: a right quote with a
            # wrong number is still evidence, just misattributed.
            cited = hits[c["passage"] - 1 : c["passage"]] if c["passage"] >= 1 else []
            source = next((h for h in [*cited, *hits] if grounded(c["quote"], h["text"])), None)
            # A statement must be a sentence, not a bare citation marker like "[3]".
            if source is None or len(re.findall(r"[A-Za-z]{2,}", c["statement"])) < 4:
                dropped += 1
            else:
                claims.append(Claim(c["statement"], c["quote"], source))
        if claims:
            kept = verify(question, claims)
            dropped += len(claims) - len(kept)
            if kept:
                return Answer(question, kept, queries, dropped)
        if step + 1 < MAX_STEPS:
            titles = "; ".join(dict.fromkeys(h["title"] for h in hits))
            prompt = f"Question: {question}\nEarlier queries: {queries}\nPages found so far: {titles}"
            rewritten = ask(REWRITE_PROMPT, prompt, REWRITE_SCHEMA).get("query", queries[-1])
            if rewritten in queries:
                break  # the model has no new idea; searching again would return the same passages
            queries.append(rewritten)
    return Answer(question, [], queries, dropped)
