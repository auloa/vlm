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


def _flatten_sub_names(name: str, sub) -> str:
    """Fold modifier sub-items into the parent name.
    """
    if isinstance(sub, dict):
        sub = [sub]
    if not isinstance(sub, list):
        return name

    sub_names = [
        _to_str(s.get("nm"))
        for s in sub
        if isinstance(s, dict) and s.get("nm")
    ]

    if not sub_names:
        return name

    return f"{name} ({', '.join(sub_names)})"


def parse_ground_truth(gt_string: str) -> dict:
    """Normalize a CORD ground_truth string into the training schema.

    Schema:
        {
          "line_items": [{"name": str, "count": str, "price": str}, ...],
          "total": str
        }

    This function does only structure normalization. It does not decide
    whether a sample is usable - the dataset class filters on the result.

    Normalizations:
    - `menu` may be a list (multi-item receipt) or a single dict
      (single-item receipt). Both become a list.
    - Modifier `sub` items are flattened into the parent `name`.
    - `count` is emitted only when the receipt provides one. We don't
      default it, since the model is learning a visual-to-text mapping
      and inventing a "1" the image doesn't show would be a hallucination.
    - Items missing both `name` and `price` are dropped.

    Raises:
        RuntimeError: the JSON is unparseable or the top-level structure
            is missing `gt_parse`. The dataset counts these separately.
    """
    try:
        gt = json.loads(gt_string)
        gt_parse = gt["gt_parse"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not parse ground truth string: {gt_string}") from exc

    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        menu = []

    line_items: list[dict[str, str]] = []

    for item in menu:
        if not isinstance(item, dict):
            continue

        name = _to_str(item.get("nm"))
        price = _to_str(item.get("price"))

        if not name and not price:
            continue

        name = _flatten_sub_names(name, item.get("sub"))
        count = _to_str(item.get("cnt"))

        # Build dict with stable key order: name, [count], price.
        item_out: dict[str, str] = {"name": name}
        if count:
            item_out["count"] = count
        item_out["price"] = price

        line_items.append(item_out)

    total_block = gt_parse.get("total") or {}
    total = _to_str(total_block.get("total_price")) if isinstance(total_block, dict) else ""

    return {"line_items": line_items, "total": total}


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

    Filter counters (all exposed for logging):
        num_loaded         total samples in the raw HF split
        num_parse_failed   raised in parse_ground_truth
        num_empty_items    parsed but produced zero line_items
        num_missing_total  parsed but produced no total
        num_too_long       tokenized target exceeded max_target_length
        num_after_filtering kept samples after max_samples cap
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

        for idx in range(len(raw)):
            item = cast(CORDRow, raw[idx])
            image = item["image"].convert("RGB")

            try:
                parsed = parse_ground_truth(item["ground_truth"])
            except RuntimeError:
                self.num_parse_failed += 1
                continue

            if not parsed["line_items"]:
                self.num_empty_items += 1
                continue

            if not parsed["total"]:
                self.num_missing_total += 1
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