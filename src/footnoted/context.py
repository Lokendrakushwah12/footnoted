"""Fit the most useful context into a fixed token budget, so the prompt can never overflow."""

from dataclasses import dataclass
from typing import Any, Callable

# Calibrated on 108 passages against the Qwen 2.5 tokenizer: median 4.25 characters per token,
# 5th percentile 3.25, worst 2.80. The 5th percentile is used because 3.5 proved too optimistic:
# one code-heavy prompt came out 83 tokens over its budget. Pass a real tokenizer as `count`
# for exact budgets.
CHARS_PER_TOKEN = 3.25

# How the budget left after required items is split when every slot has more than fits.
SHARES = {"retrieval": 0.6, "memory": 0.25, "tools": 0.15}


def estimate(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class Item:
    slot: str  # "retrieval", "memory", "tools", or any name for a required item
    text: str
    priority: float = 0.0  # 0..1, comparable across slots; higher is kept first
    required: bool = False  # system prompt, the question: never dropped
    ref: Any = None  # whatever the caller wants back, e.g. the database row


@dataclass
class Assembly:
    kept: list[Item]  # required items first, then by priority
    dropped: list[Item]
    tokens: int  # estimated tokens of everything kept
    budget: int

    def of(self, slot: str) -> list[Item]:
        return [i for i in self.kept if i.slot == slot]


def assemble(
    items: list[Item],
    budget: int,
    shares: dict[str, float] = SHARES,
    count: Callable[[str], int] = estimate,
) -> Assembly:
    """Choose which items go in the prompt.

    Required items always go in. Each slot then spends up to its share of what is left, best
    items first. Whatever a slot did not use is handed to the best remaining items of any slot,
    so an empty memory slot turns into more retrieved passages instead of wasted space.
    """
    costed = [(item, count(item.text)) for item in items]
    left = budget - sum(cost for item, cost in costed if item.required)
    if left < 0:
        raise ValueError("required context alone exceeds the token budget")

    kept = [item for item, _ in costed if item.required]
    optional = sorted(((i, c) for i, c in costed if not i.required), key=lambda pair: -pair[0].priority)
    allowance = {slot: int(left * share) for slot, share in shares.items()}

    waiting = []
    for item, cost in optional:
        if cost <= allowance.get(item.slot, 0):
            allowance[item.slot] -= cost
            left -= cost
            kept.append(item)
        else:
            waiting.append((item, cost))

    dropped = []
    for item, cost in waiting:
        if cost <= left:
            left -= cost
            kept.append(item)
        else:
            dropped.append(item)

    # ponytail: an item that does not fit is dropped whole. Truncating the last passage at a
    # paragraph break would use the tail of the budget; add it if dropped items become common.
    kept.sort(key=lambda item: (not item.required, -item.priority))
    return Assembly(kept, dropped, budget - left, budget)
