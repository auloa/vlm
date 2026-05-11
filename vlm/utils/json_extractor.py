import json


def extract_json_object(text: str) -> str | None:
    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        return None

    candidate = text[start : end + 1]

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def parse_json_object(text: str):
    extracted = extract_json_object(text)

    if extracted is None:
        return None

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None