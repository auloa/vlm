import json
from typing import Any, TypedDict, cast

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _to_str(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def _clean_dict(obj) -> dict:
    """Drop keys whose values are empty strings or None.

    Preserves nested structure: keys mapping to dicts/lists are recursed into,
    and only retained if the nested result is non-empty.
    """
    if not isinstance(obj, dict):
        return {}

    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            cleaned = _clean_dict(v)
            if cleaned:
                out[k] = cleaned
        elif isinstance(v, list):
            cleaned_list = _clean_list(v)
            if cleaned_list:
                out[k] = cleaned_list
        else:
            s = _to_str(v)
            if s:
                out[k] = s
    return out


def _clean_list(items) -> list:
    """Walk a list, cleaning each dict member and dropping anything empty."""
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if isinstance(item, dict):
            cleaned = _clean_dict(item)
            if cleaned:
                out.append(cleaned)
        elif isinstance(item, list):
            cleaned = _clean_list(item)
            if cleaned:
                out.append(cleaned)
        else:
            s = _to_str(item)
            if s:
                out.append(s)
    return out


def parse_ground_truth(gt_string: str) -> dict:
    """Normalize a CORD ground_truth string into the training schema.

    Strategy: keep CORD's native gt_parse structure verbatim. The only
    transformations are:

    - `menu` is always a list (single-item receipts encode it as a dict)
    - empty string values and empty containers are dropped
    - top-level keys passed through as-is (menu, sub_total, total, void_menu)
    - per-item keys passed through as-is (nm, cnt, price, sub, discountprice,
      num, unitprice, ...)

    This means the model is trained to extract every value visible on the
    receipt with CORD's native nomenclature. The model never has to learn
    "ignore this visible region because the schema has no place for it,"
    which weakens the visual grounding prior.

    Raises:
        RuntimeError: the JSON is unparseable or missing `gt_parse`.
    """
    try:
        gt = json.loads(gt_string)
        gt_parse = gt["gt_parse"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not parse ground truth string: {gt_string}") from exc

    out: dict[str, Any] = {}

    # menu: normalize to list, keep item keys verbatim
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        menu = []
    cleaned_menu = _clean_list(menu)
    if cleaned_menu:
        out["menu"] = cleaned_menu

    # sub_total: keep all keys verbatim
    sub_total = gt_parse.get("sub_total")
    if isinstance(sub_total, dict):
        cleaned = _clean_dict(sub_total)
        if cleaned:
            out["sub_total"] = cleaned

    # total: keep all keys verbatim
    total = gt_parse.get("total")
    if isinstance(total, dict):
        cleaned = _clean_dict(total)
        if cleaned:
            out["total"] = cleaned

    # void_menu: rare but visible on the receipt when present
    void_menu = gt_parse.get("void_menu")
    if void_menu:
        if isinstance(void_menu, dict):
            void_menu = [void_menu]
        cleaned = _clean_list(void_menu)
        if cleaned:
            out["void_menu"] = cleaned

    return out


def is_usable(parsed: dict) -> tuple[bool, str | None]:
    """Decide whether a parsed sample is useful for training."""
    menu = parsed.get("menu")
    if not menu:
        return False, "empty_items"

    total = parsed.get("total")
    if not isinstance(total, dict) or not total.get("total_price"):
        return False, "missing_total"

    return True, None


class CORDRow(TypedDict):
    image: Image.Image
    ground_truth: str


class CORDDataset(Dataset):
    """CORD receipt dataset with native CORD schema preserved.

    Each sample contains:
        image: RGB PIL image
        label: JSON string mirroring CORD's gt_parse

    If tokenizer and max_target_length are provided, targets too long are
    filtered out before max_samples is applied.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = 400,
        dataset_name: str = "naver-clova-ix/cord-v2",
        prefer_high_resolution: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_target_length: int | None = None,
    ):
        raw = load_dataset(dataset_name, split=split)

        candidates: list[dict[str, Any]] = []

        self.num_loaded = len(raw)
        self.num_parse_failed = 0
        self.num_empty_items = 0
        self.num_missing_total = 0
        self.num_too_long = 0
        self.dropped: list[dict[str, Any]] = []

        for idx in range(len(raw)):
            item = cast(CORDRow, raw[idx])
            image = item["image"].convert("RGB")

            try:
                parsed = parse_ground_truth(item["ground_truth"])
            except RuntimeError:
                self.num_parse_failed += 1
                self.dropped.append({
                    "idx": idx,
                    "reason": "parse_failed",
                    "ground_truth": item["ground_truth"],
                })
                continue

            ok, reason = is_usable(parsed)
            if not ok:
                if reason == "empty_items":
                    self.num_empty_items += 1
                elif reason == "missing_total":
                    self.num_missing_total += 1
                self.dropped.append({
                    "idx": idx,
                    "reason": reason,
                    "ground_truth": item["ground_truth"],
                })
                continue

            label = json.dumps(parsed, ensure_ascii=False)

            target_len = None
            if tokenizer is not None and max_target_length is not None:
                eos = tokenizer.eos_token or ""
                tokenized = tokenizer(
                    label + eos,
                    add_special_tokens=False,
                )
                target_len = len(tokenized["input_ids"])

                if target_len > max_target_length:
                    self.num_too_long += 1
                    self.dropped.append({
                        "idx": idx,
                        "reason": "too_long",
                        "target_len": target_len,
                        "ground_truth": item["ground_truth"],
                    })
                    continue

            width, height = image.size
            area = width * height

            candidates.append(
                {
                    "image": image,
                    "label": label,
                    "area": area,
                    "width": width,
                    "height": height,
                    "target_len": target_len,
                }
            )

        if prefer_high_resolution:
            candidates.sort(key=lambda x: x["area"], reverse=True)

        if max_samples is not None:
            candidates = candidates[:max_samples]

        self.samples = [
            {"image": item["image"], "label": item["label"]}
            for item in candidates
        ]

        self.num_after_filtering = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Image.Image | str]:
        return self.samples[index]

    def print_drop_summary(self, max_per_reason: int = 5) -> None:
        """Print a summary of dropped samples grouped by reason."""
        from collections import defaultdict
        by_reason: dict[str, list[dict]] = defaultdict(list)
        for entry in self.dropped:
            by_reason[entry["reason"]].append(entry)

        print(f"\n=== dataset filter summary ===")
        print(f"loaded:           {self.num_loaded}")
        print(f"parse_failed:     {self.num_parse_failed}")
        print(f"empty_items:      {self.num_empty_items}")
        print(f"missing_total:    {self.num_missing_total}")
        print(f"too_long:         {self.num_too_long}")
        print(f"kept after cap:   {len(self.samples)}")

        for reason, entries in by_reason.items():
            print(f"\n--- {reason} ({len(entries)} samples) ---")
            for entry in entries[:max_per_reason]:
                idx = entry["idx"]
                gt = entry["ground_truth"]
                preview = gt[:200] + ("..." if len(gt) > 200 else "")
                print(f"  idx={idx}: {preview}")
            if len(entries) > max_per_reason:
                print(f"  ... and {len(entries) - max_per_reason} more")