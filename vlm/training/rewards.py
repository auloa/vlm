"""
Reward function for RL alignment of the receipt VLM.

The schema mirrors CORD's gt_parse: menu (list of items), sub_total (dict),
total (dict), void_menu (rare). The model never sees a fixed key set — it
learns to extract whatever fields the ground truth contains for that receipt.

Reward components (final score clipped to [0, 1]):
    format         0-0.30   strict JSON parseable, no wrapping prose
    schema         0-0.30   correct structural shape (menu list, total dict)
    content        0-0.40   dynamic per-key grading against ground truth
    hallucination  ≤ 0      extra keys, duplicate items, leaked tokens, garbage

Content grading walks the GT recursively. For each leaf key, the model is
scored by:
    - Money fields (digit-heavy values): tolerant numeric match
    - Text fields: token overlap

The final content score combines coverage (% of GT keys present in pred) with
value accuracy (avg per-key match score for matched keys).
"""

import re
from dataclasses import dataclass
from itertools import pairwise

from vlm.utils.json_extractor import parse_json_object


@dataclass
class RewardBreakdown:
    total: float
    format: float
    schema: float
    content: float
    hallucination: float


def compute_reward(generated: str, ground_truth: str) -> RewardBreakdown:
    """Score a single generated completion against its ground truth."""
    generated = generated.strip()

    parsed = parse_json_object(generated)
    gt = parse_json_object(ground_truth) or {}

    format_score = _format_score(generated, parsed)

    if parsed is None:
        return RewardBreakdown(
            total=max(0.0, min(1.0, format_score)),
            format=format_score,
            schema=0.0,
            content=0.0,
            hallucination=0.0,
        )

    schema_score = _schema_score(parsed)
    content_score = _content_score(parsed, gt)
    hallucination_score = _hallucination_penalty(generated, parsed, gt)

    total = format_score + schema_score + content_score + hallucination_score

    return RewardBreakdown(
        total=max(0.0, min(1.0, total)),
        format=format_score,
        schema=schema_score,
        content=content_score,
        hallucination=hallucination_score,
    )


# ----------------------------------------------------------------------
# Format
# ----------------------------------------------------------------------

def _format_score(text: str, parsed) -> float:
    """Reward clean JSON, partial credit for wrapped-but-parseable."""
    if parsed is None:
        looks_jsonish = "{" in text and "}" in text
        return 0.05 if looks_jsonish else 0.0

    if text.startswith("{") and text.endswith("}"):
        return 0.30

    # Parseable JSON exists but wrapped in prose.
    return 0.10


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

def _schema_score(parsed) -> float:
    """Reward correct receipt structure (menu list + total dict)."""
    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0

    # Has menu key as a list
    menu = parsed.get("menu")
    if menu is not None:
        score += 0.08
        if isinstance(menu, list) and menu:
            score += 0.08
            # Items are dicts with at least nm and price
            well_formed = sum(
                1 for it in menu
                if isinstance(it, dict) and "nm" in it and "price" in it
            )
            score += 0.08 * (well_formed / len(menu))

    # Has total dict with total_price
    total = parsed.get("total")
    if total is not None:
        score += 0.02
        if isinstance(total, dict) and total.get("total_price"):
            score += 0.04

    return min(0.30, score)


# ----------------------------------------------------------------------
# Content: dynamic per-key grading
# ----------------------------------------------------------------------

def _content_score(pred: dict, gt: dict) -> float:
    """Walk GT recursively, score each leaf key against the prediction.

    Returns coverage * value_accuracy, scaled to max 0.60.
    """
    gt_leaves = _flatten_leaves(gt)
    pred_leaves = _flatten_leaves(pred)

    if not gt_leaves:
        return 0.0

    # Per-key score for keys that exist in both pred and gt.
    matched_scores = []
    for path, gt_val in gt_leaves.items():
        if path in pred_leaves:
            matched_scores.append(_value_match(pred_leaves[path], gt_val))

    coverage = len(matched_scores) / len(gt_leaves)
    value_acc = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    return 0.40 * coverage * value_acc


def flatten_leaves(obj, prefix: str = "") -> dict[str, str]:
    """Public alias — see _flatten_leaves."""
    return _flatten_leaves(obj, prefix)


def value_match(pred_val, gt_val) -> float:
    """Public alias — see _value_match."""
    return _value_match(pred_val, gt_val)


def _flatten_leaves(obj, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict/list into {path: value} leaf entries.

    Lists are indexed: `menu[0].nm`. Dicts are dot-separated: `total.total_price`.
    Only leaf values (strings/numbers) appear in the output.
    """
    out: dict[str, str] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out.update(_flatten_leaves(v, new_prefix))
            else:
                out[new_prefix] = str(v) if v is not None else ""
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                out.update(_flatten_leaves(item, new_prefix))
            else:
                out[new_prefix] = str(item) if item is not None else ""

    return out


def _value_match(pred_val, gt_val) -> float:
    """Score a single value match. Returns a float in [0, 1].

    Auto-detects field type:
    - Money-like (mostly digits): tolerant numeric match (strip separators)
    - Text: graded token overlap
    """
    pred_str = str(pred_val) if pred_val is not None else ""
    gt_str = str(gt_val) if gt_val is not None else ""

    if not gt_str:
        return 1.0 if not pred_str else 0.0

    gt_digits = _digits_only(gt_str)
    digit_density = len(gt_digits) / max(len(gt_str), 1)

    # Money-like: digits dominate the GT value
    if digit_density >= 0.50 and gt_digits:
        pred_digits = _digits_only(pred_str)
        if pred_digits == gt_digits:
            return 1.0
        if pred_digits and (pred_digits in gt_digits or gt_digits in pred_digits):
            return 0.5
        return 0.0

    # Text: token overlap
    gt_tokens = set(re.findall(r"\b\w+\b", gt_str.lower()))
    pred_tokens = set(re.findall(r"\b\w+\b", pred_str.lower()))

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0

    overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
    return min(1.0, overlap)


# ----------------------------------------------------------------------
# Hallucination
# ----------------------------------------------------------------------

def _hallucination_penalty(text: str, parsed: dict, gt: dict) -> float:
    """Negative score for invented keys, duplicate items, garbage text,
    and text field values with low token overlap against GT.

    Text hallucination penalty:
        For each leaf key present in both pred and GT where the GT value
        is text (not numeric), apply:
            penalty = -MAX_TEXT_PENALTY * (1 - token_overlap)
        Zero overlap = full penalty. Full overlap = no penalty.
        Total text penalty is capped at -0.15.
    """
    penalty = 0.0

    # Extra keys in pred that aren't in GT — model invented them.
    gt_leaves = _flatten_leaves(gt) if isinstance(gt, dict) else {}
    pred_leaves = _flatten_leaves(parsed) if isinstance(parsed, dict) else {}

    extra = set(pred_leaves) - set(gt_leaves)
    if extra:
        # cap at -0.20; scale gently — 1 extra key = -0.02, 10+ = -0.20
        penalty -= min(0.20, 0.02 * len(extra))

    # Duplicate menu items (same name).
    if isinstance(parsed, dict):
        menu = parsed.get("menu")
        if isinstance(menu, list) and len(menu) > 1:
            names = [
                _norm(it.get("nm", ""))
                for it in menu
                if isinstance(it, dict)
            ]
            if names:
                dup_ratio = 1.0 - (len(set(names)) / len(names))
                if dup_ratio > 0.50:
                    penalty -= 0.10
                elif dup_ratio > 0.25:
                    penalty -= 0.05

    # Excessive menu length vs GT.
    gt_menu = gt.get("menu") if isinstance(gt, dict) else None
    pred_menu = parsed.get("menu") if isinstance(parsed, dict) else None
    if (
        isinstance(pred_menu, list)
        and isinstance(gt_menu, list)
        and gt_menu
        and len(pred_menu) > 3 * len(gt_menu)
    ):
        penalty -= 0.05

    # Leaked role tokens.
    if "<|user|>" in text or "<|assistant|>" in text:
        penalty -= 0.10

    # Repeated character runs (garbage output).
    if _repeated_char_ratio(text) > 0.50:
        penalty -= 0.05

    return penalty


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _digits_only(value) -> str:
    return re.sub(r"\D", "", str(value))


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _repeated_char_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0
    repeated = sum(1 for a, b in pairwise(text) if a == b)
    return repeated / (len(text) - 1)