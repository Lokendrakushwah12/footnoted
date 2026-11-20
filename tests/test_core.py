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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
