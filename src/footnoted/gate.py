"""Trajectory eval gate.

record  runs the answer loop on every eval question with the real model, grades each step,
        and writes evals/scorecard.json.
check   needs no model, so it runs in CI. It fails when the scorecard is stale, when a
        trajectory took an illegal path, or when the scorecard has regressed against the golden
        run in evals/golden.json.
accept  promotes the scorecard to golden: a deliberate change that shows up in the diff.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

from .llm import ask

EVAL_FILES = [Path("evals/python.jsonl"), Path("evals/docs.jsonl"), Path("evals/unanswerable.jsonl")]
SCORECARD, GOLDEN = Path("evals/scorecard.json"), Path("evals/golden.json")
# The files that can change a trajectory. Editing any of them makes the scorecard stale.
SOURCES = [Path("src/footnoted") / name for name in ("answer.py", "context.py", "gate.py", "index.py", "llm.py")]

# The legal moves of the answer loop. Every trajectory must be a path through this graph that
# starts at retrieve and ends at answer or decline.
NEXT = {
    "retrieve": {"draft"},
    "draft": {"ground"},
    "ground": {"verify", "rewrite", "decline"},
    "verify": {"answer", "rewrite", "decline"},
    "rewrite": {"retrieve"},
    "answer": set(),
    "decline": set(),
}

# How far each headline count may fall below golden before the gate fails. Recordings of the
# same code differed by one question (the model is not fully deterministic at temperature 0),
# so the band is twice that. A deliberate break, one passage instead of five, moved right_page
# by 12 and correct by 14.
TOLERANCE = {"right_page": 2, "correct": 2, "declined": 1}

# The judge is reported, never gated on: it agreed with the key-term check on only 26 of 34
# answers, rejecting 7 that named the right API. Gate on it only if that agreement improves.

JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}
JUDGE_PROMPT = """You grade an answer to a question about software documentation. Reply \
correct = true only if the answer directly and correctly says how to do what was asked. An answer \
about a different function, module or topic, or one that never says how, is not correct."""


def valid(path: list[str]) -> bool:
    return (
        bool(path)
        and path[0] == "retrieve"
        and not NEXT.get(path[-1], {"?"})
        and all(b in NEXT.get(a, ()) for a, b in zip(path, path[1:]))
    )


def fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(SOURCES + EVAL_FILES):
        digest.update(path.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def judge(question: str, answer: str) -> bool:
    return bool(ask(JUDGE_PROMPT, f"Question: {question}\n\nAnswer: {answer}", JUDGE_SCHEMA).get("correct"))


def grade(case: dict, result) -> dict:
    """Step-level grades for one question."""
    trace = result.trace
    count = lambda step, key: sum(s[key] for s in trace if s["step"] == step)
    shown = {doc for s in trace if s["step"] == "retrieve" for doc in s["docs"]}
    text = " ".join(c.statement for c in result.claims)
    terms = case.get("terms")
    return {
        "q": case["q"],
        "answerable": bool(case["docs"]),
        "path": ">".join(s["step"] for s in trace),
        "retrieved": bool(shown & set(case["docs"])),  # retrieve: an expected page reached the model
        "drafted": count("draft", "claims"),  # draft: claims the model wrote
        "grounded": count("ground", "kept"),  # ground: claims whose quote was found
        "kept": len(result.claims),  # verify: claims that survived the check
        "answered": bool(result.claims),
        "right_page": bool({c.source["doc"] for c in result.claims} & set(case["docs"])),
        # A proxy for correctness: the answer names one of the APIs a right answer must name.
        "correct": any(t.lower() in text.lower() for t in terms) if terms and result.claims else None,
        "answer": text,
    }


def summarize(rows: list[dict]) -> dict:
    yes = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    keyed = [r for r in yes if r["correct"] is not None]  # answered, and key terms exist
    unkeyed = [r for r in yes if r["answered"] and r["correct"] is None]
    return {
        "answerable": len(yes),
        "retrieved": sum(r["retrieved"] for r in yes),
        "answered": sum(r["answered"] for r in yes),
        "right_page": sum(r["right_page"] for r in yes),
        "keyed": len(keyed),
        "correct": sum(r["correct"] for r in keyed),
        "unanswerable": len(no),
        "declined": sum(not r["answered"] for r in no),
        "claims_drafted": sum(r["drafted"] for r in rows),
        "claims_grounded": sum(r["grounded"] for r in rows),
        "claims_kept": sum(r["kept"] for r in rows),
        "invalid_paths": sum(not valid(r["path"].split(">")) for r in rows),
        # Judge calibration: the model's verdicts against the key-term verdicts.
        "judge_agrees": sum(r["judged"] == r["correct"] for r in keyed),
        "judge_yes_terms_no": sum(r["judged"] and not r["correct"] for r in keyed),
        "judge_no_terms_yes": sum(not r["judged"] and r["correct"] for r in keyed),
        "unkeyed": len(unkeyed),
        "unkeyed_judged_correct": sum(r["judged"] for r in unkeyed),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
    }


def record(index) -> dict:
    from . import llm
    from .answer import answer

    cases = [json.loads(line) for path in EVAL_FILES for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        start = time.perf_counter()
        row = grade(case, answer(index, case["q"], case["corpus"]))
        row["seconds"] = round(time.perf_counter() - start, 1)
        row["judged"] = judge(row["q"], row["answer"]) if row["answered"] else None
        rows.append(row)
        print(".", end="", flush=True)
    print()
    card = {"fingerprint": fingerprint(), "model": llm.MODEL, "summary": summarize(rows), "cases": rows}
    SCORECARD.write_text(json.dumps(card, indent=1) + "\n")
    return card


def report(card: dict) -> str:
    s = card["summary"]
    return "\n".join(
        [
            f"answerable questions: {s['answerable']}",
            f"  an expected page reached the model: {s['retrieved']}",
            f"  answered: {s['answered']}",
            f"  answered citing an expected page: {s['right_page']}",
            f"  correct by key term: {s['correct']} of the {s['keyed']} answered questions that have key terms",
            f"unanswerable questions: {s['unanswerable']}, declined: {s['declined']}",
            f"claims: {s['claims_drafted']} drafted, {s['claims_grounded']} with a verified quote, "
            f"{s['claims_kept']} kept after the check",
            f"trajectories off the legal graph: {s['invalid_paths']}",
            f"judge agrees with key terms on {s['judge_agrees']} of {s['keyed']} answers "
            f"(judge yes / terms no: {s['judge_yes_terms_no']}, judge no / terms yes: {s['judge_no_terms_yes']})",
            f"judge accepts {s['unkeyed_judged_correct']} of the {s['unkeyed']} answers that have no key terms",
            f"median seconds per question: {s['median_seconds']}",
        ]
    )


def regressions(card: dict, golden: dict) -> list[str]:
    """Questions that got worse than in the golden run."""
    before = {r["q"]: r for r in golden["cases"]}
    worse = []
    for now in card["cases"]:
        was = before.get(now["q"])
        if was is None:
            continue
        if was["right_page"] and not now["right_page"]:
            worse.append(f"no longer cites an expected page: {now['q']}")
        elif was["correct"] and not now["correct"]:
            worse.append(f"no longer correct by key term: {now['q']}")
        elif not now["answerable"] and now["answered"] and not was["answered"]:
            worse.append(f"now answers an unanswerable question: {now['q']}")
    return worse


def check() -> list[str]:
    """Reasons the gate fails. Empty means pass."""
    if not SCORECARD.exists() or not GOLDEN.exists():
        return ["no scorecard or no golden run: run `footnoted gate record`, then `footnoted gate accept`"]
    card, golden = json.loads(SCORECARD.read_text()), json.loads(GOLDEN.read_text())
    problems = []
    if card["fingerprint"] != fingerprint():
        problems.append("the scorecard is stale: the answer loop or the evals changed since it was "
                        "recorded. Run `footnoted gate record` and commit evals/scorecard.json")
    now, was = card["summary"], golden["summary"]
    if now["invalid_paths"]:
        bad = [r["path"] for r in card["cases"] if not valid(r["path"].split(">"))]
        problems.append(f"{now['invalid_paths']} trajectories left the legal graph, e.g. {bad[0]}")
    for key, slack in TOLERANCE.items():
        if now[key] < was[key] - slack:
            problems.append(f"{key} fell from {was[key]} to {now[key]}, more than the {slack} allowed")
    return problems
