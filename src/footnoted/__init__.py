"""footnoted: retrieval over documentation that can always say where an answer came from."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="footnoted")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="build the index from corpora/")

    search = sub.add_parser("search", help="search the index")
    search.add_argument("query")
    search.add_argument("--corpus", choices=["python", "mcp", "claude"])
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid", "rerank"], default="hybrid")
    search.add_argument("-k", type=int, default=5)

    evals = sub.add_parser("eval", help="score every search mode on eval files")
    evals.add_argument("files", nargs="+", type=Path)
    evals.add_argument("--across", action="store_true", help="search all corpora, not just the question's own")
    evals.add_argument("-v", "--verbose", action="store_true", help="list the missed questions")

    question = sub.add_parser("ask", help="answer a question with citations, or decline")
    question.add_argument("question")
    question.add_argument("--corpus", choices=["python", "mcp", "claude"])

    gate = sub.add_parser("gate", help="trajectory eval gate for the answer loop")
    gate.add_argument(
        "action",
        choices=["record", "check", "accept"],
        help="record: run the evals with the local model; check: compare with golden, no model "
        "needed; accept: make the current scorecard the golden run",
    )

    chat = sub.add_parser("chat", help="ask follow-up questions; remembers across sessions")
    chat.add_argument("--corpus", choices=["python", "mcp", "claude"])

    remember = sub.add_parser("eval-memory", help="score follow-up rewriting with memory (needs the local model)")
    remember.add_argument("file", type=Path)
    remember.add_argument("-v", "--verbose", action="store_true", help="show every rewritten question")

    args = parser.parse_args()
    if args.cmd == "gate" and args.action != "record":  # no model, no index: this is what CI runs
        import json
        import sys

        from . import gate as g

        if args.action == "accept":
            g.GOLDEN.write_text(g.SCORECARD.read_text())
            print(f"{g.SCORECARD} is now the golden run")
            return
        problems = g.check()
        if g.SCORECARD.exists() and g.GOLDEN.exists():
            card, golden = json.loads(g.SCORECARD.read_text()), json.loads(g.GOLDEN.read_text())
            print(g.report(card))
            for line in g.regressions(card, golden):
                print(f"  worse than golden: {line}")
        for problem in problems:
            print(f"GATE FAILED: {problem}")
        print("gate passed" if not problems else "")
        sys.exit(1 if problems else 0)
    # Imported here so --help does not load the models.
    from . import evals as ev
    from .index import Index, ingest

    if args.cmd == "ingest":
        print(f"indexed {ingest()} chunks")
        return
    index = Index()
    if args.cmd == "search":
        for hit in index.search(args.query, args.corpus, args.mode, args.k):
            print(f"\n[{hit['corpus']}] {hit['doc']} | {hit['heading']}\n{hit['text'][:300]}")
    elif args.cmd == "ask":
        from .answer import answer

        result = answer(index, args.question, args.corpus)
        print(result.render())
        if len(result.queries) > 1:
            print(f"\n(rewrote the query: {result.queries[1:]})")
    elif args.cmd == "chat":
        from .answer import answer
        from .memory import Memory, standalone

        memory = Memory(lambda text: next(iter(index.embedder.embed([text]))))
        print("Ask a question. Empty line or Ctrl-D to quit.")
        try:
            while typed := input("> ").strip():
                question = standalone(typed, memory)
                if question != typed:
                    print(f"(understood as: {question})")
                reply = answer(index, question, args.corpus).render()
                print(reply, end="\n\n")
                memory.add_turn(typed, reply[:400])
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            memory.flush()  # what was said this session becomes long-term memory
    elif args.cmd == "eval-memory":
        print(ev.run_memory(args.file, index, args.verbose))
    elif args.cmd == "gate":
        from . import gate as g

        print(g.report(g.record(index)))
    else:
        for path in args.files:
            print(ev.report(path, ev.run(path, index, args.across), args.verbose), end="\n\n")
