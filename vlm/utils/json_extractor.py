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


def _json_schema_flags(text: str) -> tuple[bool, bool, bool]:
    parsed = parse_json_object(text)

    json_ok = parsed is not None
    clean_json = isinstance(parsed, dict)

    schema_ok = (
        isinstance(parsed, dict)
        and isinstance(parsed.get("menu"), list)
        and len(parsed.get("menu")) > 0
        and isinstance(parsed.get("total"), dict)
        and parsed.get("total", {}).get("total_price") is not None
        and str(parsed.get("total", {}).get("total_price")).strip() != ""
    )

    return json_ok, clean_json, schema_ok