"""Three tiers of conversation memory.

working   the last few turns, verbatim, for this session only
episode   a short summary of turns that left working memory
fact      a durable statement about the user or their project

Episodes and facts live in SQLite, so they survive across sessions until their TTL runs out.
"""

import re
import sqlite3
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .context import Item, assemble
from .index import STOPWORDS
from .llm import ask

WORKING_TURNS = 4  # turns kept verbatim; older ones are compressed
TTL_DAYS = {"episode": 7, "fact": 30}  # multiplied by importance (1-5): important things live longer
DAY = 86400
# Calibrated with bge-small on question/fact pairs: matching pairs score 0.50-0.72 and unrelated
# ones 0.40-0.65, so no threshold separates them. The floor only removes the clearly unrelated;
# the top-k cut does the selecting, and the right memory was in the top 3 for every pair tried.
MIN_SIMILARITY = 0.45
DUPLICATE = 0.92  # a new memory this close to an old one of the same kind refreshes it instead
REWRITE_BUDGET = 900  # tokens of working turns and recalled memories shown to the rewriter

COMPRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "importance": {"type": "integer"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "importance": {"type": "integer"}},
                "required": ["text", "importance"],
            },
        },
    },
    "required": ["summary", "importance", "facts"],
}
REWRITE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}

COMPRESS_PROMPT = """You keep long-term memory for a documentation assistant. From the conversation \
excerpt, write:
- summary: one or two sentences on what the user asked about.
- importance: 1 to 5, how useful the summary will be in a later conversation.
- facts: durable facts about the user or their project that later questions may depend on: what \
they are building, the tools, versions and formats they use, their preferences. Each fact is one \
standalone sentence starting with "The user", with an importance of 1 to 5. Record only what the \
user said about themselves, never what the documentation says. Return no facts if there are none.

Example. Excerpt: "User: Our team writes everything in Rust. How do I parse JSON?" Reply:
{"summary": "The user asked how to parse JSON.", "importance": 2, "facts": [{"text": "The user's \
team writes everything in Rust.", "importance": 4}]}"""

REWRITE_PROMPT = """Rewrite the user's new question so it can be understood without the earlier \
conversation.
- Replace words like "it", "that", "them" and "one" with what they refer to in the conversation.
- If the question depends on something in the notes about the user (what they are building, the \
tools or formats they use), state that detail in the question.
- Keep every detail of the new question, and keep it to one question.
- If the question is already complete, return it unchanged.
- Never answer the question, and never return an earlier question in its place."""


class Memory:
    def __init__(
        self,
        embed: Callable[[str], np.ndarray],
        path: Path = Path("data/memory.db"),
        now: Callable[[], float] = time.time,
    ) -> None:
        self.embed, self.now = embed, now
        self.turns: list[tuple[str, str]] = []
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "create table if not exists memories(id integer primary key, kind, text, "
            "importance integer, created real, expires real, vector blob)"
        )
        self.evict()

    def evict(self) -> int:
        """Delete memories whose TTL has run out."""
        gone = self.db.execute("delete from memories where expires < ?", (self.now(),)).rowcount
        self.db.commit()
        return gone

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        if len(self.turns) > WORKING_TURNS:
            overflow, self.turns = self.turns[:-WORKING_TURNS], self.turns[-WORKING_TURNS:]
            self.compress(overflow)

    def flush(self) -> None:
        """End of session: move what is still in working memory into long-term memory."""
        if self.turns:
            self.compress(self.turns)
            self.turns = []

    def compress(self, turns: list[tuple[str, str]]) -> None:
        transcript = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
        reply = ask(COMPRESS_PROMPT, transcript, COMPRESS_SCHEMA)
        if reply.get("summary"):
            self.store("episode", reply["summary"], reply.get("importance", 3))
        for fact in reply.get("facts", []):
            self.store("fact", fact["text"], fact["importance"])

    def _vector(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embed(text), dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def store(self, kind: str, text: str, importance: int) -> None:
        importance = min(5, max(1, int(importance)))
        vector, now = self._vector(text), self.now()
        expires = now + TTL_DAYS[kind] * importance * DAY
        for row in self.db.execute("select id, vector, importance from memories where kind = ?", (kind,)):
            if float(np.frombuffer(row["vector"], dtype=np.float32) @ vector) >= DUPLICATE:
                # Said again: keep one copy, and let it live as long as the stronger of the two.
                self.db.execute(
                    "update memories set importance = max(importance, ?), expires = max(expires, ?) where id = ?",
                    (importance, expires, row["id"]),
                )
                break
        else:
            self.db.execute(
                "insert into memories(kind, text, importance, created, expires, vector) values (?,?,?,?,?,?)",
                (kind, text, importance, now, expires, vector.tobytes()),
            )
        self.db.commit()

    def recall(self, query: str, k: int = 3) -> list[tuple[float, sqlite3.Row]]:
        """The k live memories most relevant to the query, as (score, row). Score is similarity
        scaled by importance, so of two equally similar memories the more important one wins."""
        # ponytail: scans every memory. Fine for thousands; index the vectors beyond that.
        query_vector = self._vector(query)
        scored = []
        for row in self.db.execute("select * from memories where expires >= ?", (self.now(),)):
            similarity = float(np.frombuffer(row["vector"], dtype=np.float32) @ query_vector)
            if similarity >= MIN_SIMILARITY:
                scored.append((similarity * (0.5 + 0.1 * row["importance"]), row))
        return sorted(scored, key=lambda pair: -pair[0])[:k]


def standalone(question: str, memory: Memory) -> str:
    """Rewrite a follow-up ("what about a single file?") into a question retrieval can use."""
    recalled = memory.recall(question)
    if not memory.turns and not recalled:
        return question
    items = [Item("instructions", REWRITE_PROMPT, required=True), Item("question", question, required=True)]
    # Later turns matter more for resolving "it" and "that".
    items += [
        Item("working", f"User: {q}\nAssistant: {a}", priority=(n + 1) / len(memory.turns), ref=n)
        for n, (q, a) in enumerate(memory.turns)
    ]
    items += [Item("memory", row["text"], priority=score) for score, row in recalled]
    context = assemble(items, REWRITE_BUDGET, shares={"working": 0.6, "memory": 0.4})
    turns = "\n".join(i.text for i in sorted(context.of("working"), key=lambda i: i.ref))
    notes = "\n".join(f"- {i.text}" for i in context.of("memory"))
    prompt = (
        f"Earlier conversation:\n{turns or '(none)'}\n\n"
        f"Notes about the user:\n{notes or '(none)'}\n\nNew question: {question}"
    )
    rewritten = ask(REWRITE_PROMPT, prompt, REWRITE_SCHEMA).get("question") or question
    # A rewrite may add context but must keep what the user asked. A small model sometimes
    # returns the previous question instead; that is caught here and the typed question used.
    words = lambda text: set(re.findall(r"\w+", text.lower())) - STOPWORDS
    asked = words(question)
    return rewritten if len(asked & words(rewritten)) * 2 >= len(asked) else question
