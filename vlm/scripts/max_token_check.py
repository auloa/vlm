import json

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vlm.data.dataset import parse_ground_truth

DATASET_NAME = "naver-clova-ix/cord-v2"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLIT = "train"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ds = load_dataset(DATASET_NAME, split=SPLIT)

lengths = []

for item in ds:
    try:
        parsed = parse_ground_truth(item["ground_truth"])
        label = json.dumps(parsed, ensure_ascii=False)

        token_ids = tokenizer(
            label,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(token_ids))
    except Exception:
        continue

lengths = np.array(lengths)

print("num samples:", len(lengths))
print("min:", lengths.min())
print("mean:", lengths.mean())
print("median:", np.percentile(lengths, 50))
print("p75:", np.percentile(lengths, 75))
print("p85:", np.percentile(lengths, 85))
print("p90:", np.percentile(lengths, 90))
print("p95:", np.percentile(lengths, 95))
print("p99:", np.percentile(lengths, 99))
print("max:", lengths.max())