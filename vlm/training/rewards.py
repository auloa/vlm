"""
Reward function for RL alignment of the receipt VLM.

Public API used elsewhere:
    - compute_reward(generated, ground_truth) -> RewardBreakdown
    - flatten_leaves(obj, prefix="") -> dict[str, str]
    - value_match(pred_val, gt_val) -> float

Design:
    - Format/schema are small bonuses, not the main reward.
    - Empty valid JSON gets tiny reward.
    - Content dominates.
    - Menu is scored by approximate item matching, not only menu[0], menu[1].
    - Text values use token F1.
    - Numeric/money values use tolerant numeric matching.
"""

import json
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
    generated = generated.strip()

    parsed = parse_json_object(generated)
    gt = parse_json_object(ground_truth)

    if gt is None:
        try:
            gt = json.loads(ground_truth)
        except Exception:
            gt = {}

    if parsed is None:
        fmt = _format_score(generated, parsed)
        return RewardBreakdown(
            total=_clip(fmt),
            format=fmt,
            schema=0.0,
            content=0.0,
            hallucination=0.0,
        )

    if not isinstance(parsed, dict):
        return RewardBreakdown(
            total=0.01,
            format=0.01,
            schema=0.0,
            content=0.0,
            hallucination=0.0,
        )

    if not isinstance(gt, dict):
        gt = {}

    fmt = _format_score(generated, parsed)
    schema_bonus, schema_gate = _schema_score_and_gate(parsed, gt)

    if _is_empty_prediction(parsed, gt):
        total = _clip(fmt + min(schema_bonus, 0.01))
        return RewardBreakdown(
            total=total,
            format=fmt,
            schema=min(schema_bonus, 0.01),
            content=0.0,
            hallucination=0.0,
        )

    raw_content = _content_score(parsed, gt)
    content = schema_gate * raw_content
    hallucination = _hallucination_penalty(generated, parsed, gt)

    total = _clip(fmt + schema_bonus + content + hallucination)

    return RewardBreakdown(
        total=total,
        format=fmt,
        schema=schema_bonus,
        content=content,
        hallucination=hallucination,
    )


# ----------------------------------------------------------------------
# Format / schema
# ----------------------------------------------------------------------

def _format_score(text: str, parsed) -> float:
    """
    Tiny bonus only. Valid JSON should not dominate reward.
    """
    if parsed is None:
        return 0.005 if ("{" in text and "}" in text) else 0.0

    if text.startswith("{") and text.endswith("}"):
        return 0.03

    return 0.01


def _schema_score_and_gate(parsed: dict, gt: dict) -> tuple[float, float]:
    """
    Returns:
        schema_bonus: small positive bonus, max 0.05
        schema_gate: multiplier for content, 0-1
    """
    if not isinstance(parsed, dict):
        return 0.0, 0.0

    bonus = 0.0
    gate = 0.35

    gt_menu = gt.get("menu")
    pred_menu = parsed.get("menu")

    gt_total = gt.get("total")
    pred_total = parsed.get("total")

    if isinstance(gt_menu, list):
        if isinstance(pred_menu, list):
            bonus += 0.015
            gate += 0.25
        else:
            gate -= 0.20

    if isinstance(gt_total, dict):
        if isinstance(pred_total, dict):
            bonus += 0.015
            gate += 0.25
        else:
            gate -= 0.20

    if gt:
        overlap = len(set(parsed.keys()) & set(gt.keys())) / max(len(gt), 1)
        bonus += 0.02 * overlap
        gate += 0.15 * overlap

    return min(0.05, max(0.0, bonus)), min(1.0, max(0.0, gate))


def _is_empty_prediction(parsed: dict, gt: dict) -> bool:
    gt_leaves = _flatten_leaves(gt)
    pred_leaves = _flatten_leaves(parsed)

    if not gt_leaves:
        return False

    if not pred_leaves:
        return True

    meaningful = [
        str(v).strip()
        for v in pred_leaves.values()
        if str(v).strip() not in {"", "null", "None", "[]", "{}"}
    ]

    return len(meaningful) == 0


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------

def _content_score(pred: dict, gt: dict) -> float:
    """
    Max content reward: 0.85

    Breakdown:
        total:      0.25
        menu:       0.45
        sub_total:  0.10
        other:      0.05
    """
    if not gt:
        return 0.0

    total_score = _total_score(pred.get("total"), gt.get("total"))
    menu_score = _menu_score(pred.get("menu"), gt.get("menu"))
    subtotal_score = _generic_structure_score(pred.get("sub_total"), gt.get("sub_total"))
    other_score = _other_fields_score(pred, gt)

    score = (
        0.25 * total_score
        + 0.45 * menu_score
        + 0.10 * subtotal_score
        + 0.05 * other_score
    )

    return min(0.85, max(0.0, score))


def _total_score(pred_total, gt_total) -> float:
    if not isinstance(gt_total, dict) or not gt_total:
        return 1.0 if not pred_total else 0.0

    if not isinstance(pred_total, dict):
        return 0.0

    gt_leaves = _flatten_leaves(gt_total)
    pred_leaves = _flatten_leaves(pred_total)

    if not gt_leaves:
        return 1.0

    weighted_sum = 0.0
    weight_total = 0.0

    for path, gt_val in gt_leaves.items():
        key = path.split(".")[-1]
        weight = 3.0 if key in {"total_price", "total", "price"} else 1.0
        pred_val = pred_leaves.get(path, "")

        weighted_sum += weight * _value_match(pred_val, gt_val)
        weight_total += weight

    return weighted_sum / max(weight_total, 1e-8)


def _menu_score(pred_menu, gt_menu) -> float:
    """
    Approximate item-level menu score.
    Does not require exact item order.
    """
    if not isinstance(gt_menu, list) or len(gt_menu) == 0:
        return 1.0 if not pred_menu else 0.0

    if not isinstance(pred_menu, list) or len(pred_menu) == 0:
        return 0.0

    gt_items = [x for x in gt_menu if isinstance(x, dict)]
    pred_items = [x for x in pred_menu if isinstance(x, dict)]

    if not gt_items:
        return 1.0 if not pred_items else 0.0

    if not pred_items:
        return 0.0

    matches = _greedy_item_matches(pred_items, gt_items)

    if not matches:
        return 0.0

    real_matches = [m for m in matches if m[2] >= 0.45]

    precision = len(real_matches) / len(pred_items)
    recall = len(real_matches) / len(gt_items)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    quality = sum(m[2] for m in real_matches) / len(real_matches) if real_matches else 0.0

    return min(1.0, max(0.0, 0.65 * f1 + 0.35 * quality))


def _greedy_item_matches(pred_items: list[dict], gt_items: list[dict]) -> list[tuple[int, int, float]]:
    candidates = []

    for pi, pred in enumerate(pred_items):
        for gi, gt in enumerate(gt_items):
            candidates.append((pi, gi, _item_similarity(pred, gt)))

    candidates.sort(key=lambda x: x[2], reverse=True)

    used_pred = set()
    used_gt = set()
    matches = []

    for pi, gi, sim in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        if sim <= 0.0:
            continue

        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, sim))

    return matches


def _item_similarity(pred_item: dict, gt_item: dict) -> float:
    pred_name = pred_item.get("nm", "")
    gt_name = gt_item.get("nm", "")

    pred_price = pred_item.get("price", "")
    gt_price = gt_item.get("price", "")

    name_score = _text_f1(pred_name, gt_name)
    price_score = _money_match(pred_price, gt_price)

    extra_scores = []
    for key, gt_val in gt_item.items():
        if key in {"nm", "price"}:
            continue
        if key in pred_item:
            extra_scores.append(_value_match(pred_item[key], gt_val))

    extra_score = sum(extra_scores) / len(extra_scores) if extra_scores else 0.0

    if str(gt_name).strip() and str(gt_price).strip():
        return 0.55 * name_score + 0.35 * price_score + 0.10 * extra_score

    if str(gt_name).strip():
        return 0.85 * name_score + 0.15 * extra_score

    if str(gt_price).strip():
        return 0.85 * price_score + 0.15 * extra_score

    return extra_score


def _generic_structure_score(pred_obj, gt_obj) -> float:
    if gt_obj is None:
        return 1.0 if pred_obj is None else 0.0

    gt_leaves = _flatten_leaves(gt_obj)
    pred_leaves = _flatten_leaves(pred_obj)

    if not gt_leaves:
        return 1.0 if not pred_leaves else 0.0

    matched = [
        _value_match(pred_leaves[path], gt_val)
        for path, gt_val in gt_leaves.items()
        if path in pred_leaves
    ]

    coverage = len(matched) / len(gt_leaves)
    value_acc = sum(matched) / len(matched) if matched else 0.0

    return coverage * value_acc


def _other_fields_score(pred: dict, gt: dict) -> float:
    excluded = {"menu", "total", "sub_total"}

    gt_other = {k: v for k, v in gt.items() if k not in excluded}
    pred_other = {k: v for k, v in pred.items() if k not in excluded}

    if not gt_other:
        return 1.0 if not pred_other else 0.0

    return _generic_structure_score(pred_other, gt_other)


# ----------------------------------------------------------------------
# Value matching
# ----------------------------------------------------------------------

def value_match(pred_val, gt_val) -> float:
    return _value_match(pred_val, gt_val)


def _value_match(pred_val, gt_val) -> float:
    pred_str = "" if pred_val is None else str(pred_val)
    gt_str = "" if gt_val is None else str(gt_val)

    if not gt_str.strip():
        return 1.0 if not pred_str.strip() else 0.0

    if _looks_numeric(gt_str):
        return _money_match(pred_str, gt_str)

    return _text_f1(pred_str, gt_str)


def _money_match(pred_val, gt_val) -> float:
    pred_num = _parse_number(pred_val)
    gt_num = _parse_number(gt_val)

    if gt_num is None:
        return _text_f1(pred_val, gt_val)

    if pred_num is None:
        pred_digits = _digits_only(pred_val)
        gt_digits = _digits_only(gt_val)

        if pred_digits and pred_digits == gt_digits:
            return 1.0

        if pred_digits and gt_digits and (pred_digits in gt_digits or gt_digits in pred_digits):
            return 0.4

        return 0.0

    diff = abs(pred_num - gt_num)

    if diff < 1e-8:
        return 1.0
    if diff <= 0.01:
        return 0.9
    if diff <= 0.05:
        return 0.7
    if diff <= 0.10:
        return 0.5
    if diff <= 0.50:
        return 0.25

    return 0.0


def _text_f1(pred_val, gt_val) -> float:
    pred_tokens = _tokens(pred_val)
    gt_tokens = _tokens(gt_val)

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0

    if not pred_tokens:
        return 0.0

    overlap = len(pred_tokens & gt_tokens)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    return 2 * precision * recall / max(precision + recall, 1e-8)


def _looks_numeric(value) -> bool:
    s = str(value).strip()
    digits = _digits_only(s)

    if not digits:
        return False

    return len(digits) / max(len(s), 1) >= 0.50


def _parse_number(value) -> float | None:
    s = str(value).strip()

    if not s:
        return None

    s = re.sub(r"[^0-9,.\-]", "", s)

    if not re.search(r"\d", s):
        return None

    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(s)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Hallucination
# ----------------------------------------------------------------------

def _hallucination_penalty(text: str, parsed: dict, gt: dict) -> float:
    penalty = 0.0

    gt_leaves = _flatten_leaves(gt)
    pred_leaves = _flatten_leaves(parsed)

    # Extra top-level sections.
    extra_top = set(parsed.keys()) - set(gt.keys())
    if extra_top:
        penalty -= min(0.15, 0.04 * len(extra_top))

    # Extra leaf paths, except menu item paths because menu is handled separately.
    extra_leaf_paths = [
        path
        for path in set(pred_leaves) - set(gt_leaves)
        if not path.startswith("menu[")
    ]

    if extra_leaf_paths:
        penalty -= min(0.20, 0.015 * len(extra_leaf_paths))

    # Wrong non-empty value where GT is empty.
    for path, gt_val in gt_leaves.items():
        if path not in pred_leaves:
            continue

        if not str(gt_val).strip() and str(pred_leaves[path]).strip():
            penalty -= 0.02

    penalty += _menu_hallucination_penalty(parsed.get("menu"), gt.get("menu"))

    if "<|user|>" in text or "<|assistant|>" in text or "<|system|>" in text:
        penalty -= 0.10

    if _repeated_char_ratio(text) > 0.50:
        penalty -= 0.05

    gt_len = len(str(gt))
    if gt_len > 0 and len(text) > 4 * gt_len:
        penalty -= 0.05

    return max(-0.35, penalty)


def _menu_hallucination_penalty(pred_menu, gt_menu) -> float:
    if not isinstance(pred_menu, list):
        return 0.0

    pred_items = [x for x in pred_menu if isinstance(x, dict)]

    if not pred_items:
        return 0.0

    if not isinstance(gt_menu, list):
        return -min(0.15, 0.03 * len(pred_items))

    gt_items = [x for x in gt_menu if isinstance(x, dict)]

    if not gt_items:
        return -min(0.15, 0.03 * len(pred_items))

    matches = _greedy_item_matches(pred_items, gt_items)
    real_matches = [m for m in matches if m[2] >= 0.45]

    unmatched_pred = max(0, len(pred_items) - len(real_matches))

    penalty = -min(0.20, 0.03 * unmatched_pred)

    if len(pred_items) > 2 * len(gt_items):
        penalty -= 0.10
    elif len(pred_items) > 1.5 * len(gt_items):
        penalty -= 0.05

    names = [
        _norm(item.get("nm", ""))
        for item in pred_items
        if _norm(item.get("nm", ""))
    ]

    if names:
        duplicate_ratio = 1.0 - (len(set(names)) / len(names))

        if duplicate_ratio > 0.50:
            penalty -= 0.10
        elif duplicate_ratio > 0.25:
            penalty -= 0.05

    return penalty


# ----------------------------------------------------------------------
# Flattening
# ----------------------------------------------------------------------

def flatten_leaves(obj, prefix: str = "") -> dict[str, str]:
    return _flatten_leaves(obj, prefix)


def _flatten_leaves(obj, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, (dict, list)):
                out.update(_flatten_leaves(value, new_prefix))
            else:
                out[new_prefix] = "" if value is None else str(value)

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            new_prefix = f"{prefix}[{index}]"

            if isinstance(item, (dict, list)):
                out.update(_flatten_leaves(item, new_prefix))
            else:
                out[new_prefix] = "" if item is None else str(item)

    return out


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def _tokens(value) -> set[str]:
    return set(re.findall(r"\b\w+\b", str(value).lower()))


def _digits_only(value) -> str:
    return re.sub(r"\D", "", str(value))


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _repeated_char_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0

    repeated = sum(1 for a, b in pairwise(text) if a == b)
    return repeated / (len(text) - 1)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))