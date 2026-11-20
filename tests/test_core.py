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
    from footnoted.context import Item, assemble, estimate

    system = Item("system", "s" * 350, required=True)  # 101 tokens
    passages = [Item("retrieval", "p" * 700, priority=1 / n) for n in range(1, 11)]  # 201 tokens each
    for budget in (150, 400, 1000, 5000):
        got = assemble([system, *passages], budget)
        assert system in got.kept and got.tokens <= budget
        assert got.tokens == sum(estimate(i.text) for i in got.kept)
        # Best passages first, none skipped.
        assert got.of("retrieval") == passages[: len(got.of("retrieval"))]
    try:
        assemble([system], budget=50)
        raise AssertionError("expected an error when required items cannot fit")
    except ValueError:
        pass


def test_assemble_shares_and_redistribution():
    from footnoted.context import Item, assemble

    passages = [Item("retrieval", "p" * 700, priority=0.9) for _ in range(10)]  # 201 tokens each
    memories = [Item("memory", "m" * 350, priority=0.1) for _ in range(10)]  # 101 tokens each
    count = lambda got: (len(got.of("retrieval")), len(got.of("memory")))
    # Shares first: retrieval's 600 tokens buy 2 passages, memory's 250 buy 2 memories, even
    # though every memory has a lower priority than every passage. The 396 tokens left over
    # (including the unused tools share) then go by priority: 1 more passage, then 1 more memory.
    assert count(assemble([*passages, *memories], budget=1000)) == (3, 3)
    # With no memories, retrieval takes the space memory would have had.
    assert count(assemble(passages, budget=1000)) == (4, 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
