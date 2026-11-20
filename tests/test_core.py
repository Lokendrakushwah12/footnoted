"""Checks for the pure logic. Run: uv run python tests/test_core.py"""

from footnoted.corpus import MAX_CHARS, pack, sections
from footnoted.index import rrf


def test_sphinx_headings():
    text = "*****\nTitle\n*****\n\nIntro.\n\nSection\n=======\n\nBody.\n\n+---+\n| a |\n+---+\n"
    got = [(h, "\n".join(b).strip()) for h, b in sections(text, markdown=False)]
    assert got == [("Title", "Intro."), ("Section", "Body.\n\n+---+\n| a |\n+---+")], got


def test_markdown_headings_ignore_code_fences():
    text = "# Title\nIntro.\n```bash\n# a comment\n```\n> ```py\n> # another\n> ```\n## Next\nBody."
    assert [h for h, _ in sections(text, markdown=True)] == ["Title", "Next"]


def test_pack_respects_limit_and_keeps_everything():
    lines = ["short para", "", "x" * 3000, "", *[f"line {i} " + "y" * 100 for i in range(40)]]
    pieces = list(pack(lines))
    assert all(len(p) <= MAX_CHARS for p in pieces), [len(p) for p in pieces]
    assert sum(p.count("x") for p in pieces) == 3000 and sum(p.count("line ") for p in pieces) == 40


def test_rrf_rewards_agreement():
    assert rrf([1, 2, 3], [3, 1, 4])[:2] == [1, 3]


def test_grounded_accepts_real_quotes_only():
    from footnoted.answer import grounded

    passage = "The module  pdb defines an interactive\nsource code debugger for Python programs."
    assert grounded("'an interactive source-code debugger'", passage)  # punctuation and whitespace ignored
    assert not grounded("active source code debugger", passage)  # must match whole words
    assert not grounded("an interactive source code profiler", passage)  # invented
    assert not grounded("pdb", passage)  # too short to count as evidence


def test_assemble_never_overflows_and_keeps_required():
    from footnoted.context import Item, assemble

    # count=len makes one character one token, so the numbers below are exact.
    system = Item("system", "s" * 100, required=True)
    passages = [Item("retrieval", "p" * 200, priority=1 / n) for n in range(1, 11)]
    for budget in (150, 400, 1000, 5000):
        got = assemble([system, *passages], budget, count=len)
        assert system in got.kept and got.tokens <= budget
        assert got.tokens == sum(len(i.text) for i in got.kept)
        # Best passages first, none skipped.
        assert got.of("retrieval") == passages[: len(got.of("retrieval"))]
    try:
        assemble([system], budget=50, count=len)
        raise AssertionError("expected an error when required items cannot fit")
    except ValueError:
        pass


def test_assemble_shares_and_redistribution():
    from footnoted.context import Item, assemble

    passages = [Item("retrieval", "p" * 200, priority=0.9) for _ in range(10)]
    memories = [Item("memory", "m" * 100, priority=0.1) for _ in range(10)]
    count = lambda items: (lambda got: (len(got.of("retrieval")), len(got.of("memory"))))(
        assemble(items, budget=1000, count=len)
    )
    # Shares first: retrieval's 600 tokens buy 3 passages and memory's 250 buy 2 memories, even
    # though every memory has a lower priority than every passage. The 200 left over (including
    # the unused tools share) go by priority, which buys a fourth passage.
    assert count([*passages, *memories]) == (4, 2)
    # With no memories, retrieval takes the space memory would have had.
    assert count(passages) == (5, 0)


def _memory(path, clock):
    """A Memory with a word-overlap embedder and a canned compressor, so no model is needed."""
    import zlib

    import numpy as np

    from footnoted import memory

    def embed(text):
        vector = np.zeros(64, dtype=np.float32)
        for word in text.lower().replace(".", "").split():
            vector[zlib.crc32(word.encode()) % 64] += 1
        return vector

    memory.ask = lambda system, user, schema: {
        "summary": "The user asked about temporary files.",
        "importance": 2,
        "facts": [{"text": "The user builds an MCP server.", "importance": 5}],
    }
    return memory.Memory(embed, path, now=lambda: clock[0])


def test_memory_tiers_persistence_and_ttl():
    import tempfile
    from pathlib import Path

    from footnoted.memory import DAY, WORKING_TURNS

    with tempfile.TemporaryDirectory() as folder:
        path, clock = Path(folder) / "memory.db", [1_000_000.0]
        first = _memory(path, clock)
        for n in range(WORKING_TURNS + 2):
            first.add_turn(f"question {n}", "answer")
        # Working memory stays bounded; what fell out became one episode and one fact. The fact
        # was extracted twice (two overflows) but stored once.
        assert [q for q, _ in first.turns] == [f"question {n}" for n in range(2, WORKING_TURNS + 2)]
        kinds = lambda m: sorted(r["kind"] for r in m.db.execute("select kind from memories"))
        assert kinds(first) == ["episode", "fact"]

        # A new session has no working memory but recalls the fact.
        second = _memory(path, clock)
        assert second.turns == []
        assert second.recall("which transport for an MCP server")[0][1]["text"] == "The user builds an MCP server."

        # Importance stretches the TTL: the episode (importance 2) lasts 14 days, the fact
        # (importance 5) lasts 150.
        clock[0] += 15 * DAY
        assert kinds(_memory(path, clock)) == ["fact"]
        clock[0] += 150 * DAY
        assert kinds(_memory(path, clock)) == []


def test_rewrite_that_drops_the_question_is_rejected():
    import tempfile
    from pathlib import Path

    from footnoted import memory

    with tempfile.TemporaryDirectory() as folder:
        m = _memory(Path(folder) / "memory.db", [0.0])
        m.add_turn("How do I read an INI configuration file?", "answer")
        typed = "How do I keep a priority queue in a plain list?"
        memory.ask = lambda *_: {"question": "How do I read an INI configuration file?"}
        assert memory.standalone(typed, m) == typed  # drifted to the earlier question: rejected
        faithful = "How do I keep a priority queue in a plain Python list?"
        memory.ask = lambda *_: {"question": faithful}
        assert memory.standalone(typed, m) == faithful  # adds context, keeps the question: accepted


def test_gate_accepts_only_legal_trajectories():
    from footnoted.gate import valid

    assert valid("retrieve>draft>ground>verify>answer".split(">"))
    assert valid("retrieve>draft>ground>rewrite>retrieve>draft>ground>decline".split(">"))
    assert not valid("retrieve>draft>ground>answer".split(">"))  # answered without the check
    assert not valid("draft>ground>verify>answer".split(">"))  # never retrieved
    assert not valid("retrieve>draft>ground>verify".split(">"))  # stopped without an outcome


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
