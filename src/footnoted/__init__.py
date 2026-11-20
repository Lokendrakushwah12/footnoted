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

    answers = sub.add_parser("eval-ask", help="score the answer loop on eval files (needs the local model)")
    answers.add_argument("files", nargs="+", type=Path)
    answers.add_argument("--limit", type=int, help="only the first N questions of each file")

    args = parser.parse_args()
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
    elif args.cmd == "eval-ask":
        print(ev.run_answers(args.files, index, args.limit))
    else:
        for path in args.files:
            print(ev.report(path, ev.run(path, index, args.across), args.verbose), end="\n\n")
