"""Load the doc sets under corpora/ and split them into chunks."""

import re
from pathlib import Path
from typing import Iterator, NamedTuple


class Chunk(NamedTuple):
    corpus: str
    doc: str  # stable page id: a relative file path or a URL path
    title: str  # page title
    heading: str  # nearest section heading above the text
    text: str


# About 300 tokens, so title + heading + text stays inside bge-small's 512-token window.
MAX_CHARS = 1200

_UNDERLINE = re.compile(r'^([=\-*^"~#+])\1{2,}$')
_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)")
_MCP_PAGE = re.compile(r"^# (?P<title>.+)\nSource: (?P<url>https://\S+)\n", re.M)
_CLAUDE_PAGE = re.compile(r"^---\ntitle: (?P<title>.*)\nurl: (?P<url>\S+)\n(?:.*\n)*?---\n", re.M)


def sections(text: str, markdown: bool) -> Iterator[tuple[str, list[str]]]:
    """Yield (heading, body lines). Handles Markdown '#' headings and Sphinx underlined ones."""
    lines = text.splitlines()
    heading, body, fenced, i = "", [], False, 0
    while i < len(lines):
        line, bare, new = lines[i], lines[i].strip(), None
        if markdown:
            if line.lstrip("> ").startswith("```"):
                fenced = not fenced  # '#' inside a code fence is a comment, not a heading
            elif not fenced and (m := _MD_HEADING.match(line)):
                new = m[1].strip()
        elif _UNDERLINE.match(bare):
            i += 1  # a title's overline, or a stray rule
            continue
        elif bare and i + 1 < len(lines):
            under = lines[i + 1].strip()
            if _UNDERLINE.match(under) and len(under) >= len(bare):
                new = bare
                i += 1
        if new is None:
            body.append(line)
        else:
            if body:
                yield heading, body
            heading, body = new, []
        i += 1
    if body:
        yield heading, body


def pack(lines: list[str]) -> Iterator[str]:
    """Group paragraphs into pieces of at most MAX_CHARS; oversized paragraphs split by line."""
    # ponytail: no overlap between chunks; the title/heading prefix carries the context.
    # Add a sliding overlap if evals show answers straddling chunk boundaries.
    piece = ""
    for para in re.split(r"\n\s*\n", "\n".join(lines)):
        if not para.strip():
            continue
        units = [para] if len(para) <= MAX_CHARS else para.splitlines()
        for n, unit in enumerate(units):
            for start in range(0, len(unit), MAX_CHARS):  # a single very long line
                part = unit[start : start + MAX_CHARS]
                if piece and len(piece) + len(part) + 2 > MAX_CHARS:  # 2 for the separator
                    yield piece
                    piece = ""
                piece += ("" if not piece else "\n\n" if n == 0 else "\n") + part
    if piece:
        yield piece


def _python(root: Path):
    for path in sorted(root.rglob("*.txt")):
        doc = path.relative_to(root).as_posix()
        # ponytail: 2.3 MB of one-line commit notes; pure noise for question answering.
        if doc != "whatsnew/changelog.txt":
            yield doc, "", path.read_text(encoding="utf-8"), False


def _llms(path: Path, page: re.Pattern, prefix: str, skip: tuple[str, ...] = ()):
    """Split an llms-full.txt dump into pages, keyed by URL path."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marks = list(page.finditer(text))
    for mark, nxt in zip(marks, marks[1:] + [None]):
        doc = mark["url"].removeprefix(prefix)
        if not doc.startswith(skip):
            yield doc, mark["title"], text[mark.end() : nxt.start() if nxt else len(text)], True


def chunks(root: Path = Path("corpora")) -> Iterator[Chunk]:
    sources = {
        "python": _python(root / "python"),
        "mcp": _llms(root / "mcp/llms-full.txt", _MCP_PAGE, "https://modelcontextprotocol.io/"),
        # ponytail: api/ is 608 auto-generated endpoint pages repeated per SDK; guides only.
        "claude": _llms(
            root / "claude/llms-full.txt",
            _CLAUDE_PAGE,
            "https://platform.claude.com/docs/en/",
            skip=("api/",),
        ),
    }
    for corpus, pages in sources.items():
        for doc, title, text, markdown in pages:
            for heading, body in sections(text, markdown):
                title = title or heading  # Python pages: the first heading is the title
                for piece in pack(body):
                    yield Chunk(corpus, doc, title, heading, piece)
