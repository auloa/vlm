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


def parse_ground_truth(gt_string: str) -> dict:
    try:
        gt = json.loads(gt_string)
        gt_parse = gt["gt_parse"]

        line_items = [
            {
                "name": _to_str(item.get("nm")),
                "count": _to_str(item.get("cnt")),
                "price": _to_str(item.get("price")),
            }
            for item in gt_parse.get("menu", [])
            if isinstance(item, dict) and (item.get("nm") or item.get("price"))
        ]

        total = _to_str(gt_parse.get("total", {}).get("total_price"))

        return {
            "line_items": line_items,
            "total": total,
        }

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not parse ground truth string: {gt_string}") from exc


class CORDRow(TypedDict):
    image: Image.Image
    ground_truth: str


class CORDDataset(Dataset):
    """CORD receipt dataset converted to simple JSON targets.

    Each sample contains:
        image: RGB PIL image
        label: JSON string with line_items and total

    If tokenizer and max_target_length are provided, targets that are too long
    are filtered out before max_samples is applied.
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
        self.num_too_long = 0
        self.num_empty_items = 0
        self.num_missing_total = 0

        # Per-sample drop log: idx, reason, raw ground_truth. Useful for
        # auditing what the parser silently rejects without re-running it.
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

            if not parsed["line_items"]:
                self.num_empty_items += 1
                self.dropped.append({
                    "idx": idx,
                    "reason": "empty_items",
                    "ground_truth": item["ground_truth"],
                })
                continue

            if not parsed["total"]:
                self.num_missing_total += 1
                self.dropped.append({
                    "idx": idx,
                    "reason": "missing_total",
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
            {
                "image": item["image"],
                "label": item["label"],
            }
            for item in candidates
        ]

        self.num_after_filtering = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Image.Image | str]:
        return self.samples[index]

    def print_drop_summary(self, max_per_reason: int = 5) -> None:
        """Print a summary of dropped samples grouped by reason.

        Useful at training startup to surface what got filtered out.
        """
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