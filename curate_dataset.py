"""Build reviewable training-data candidates from approved, licensed JSONL sources.

This program intentionally does *not* crawl the web or modify a training set.
It only accepts records that identify an entry in an explicit source allowlist,
then writes a reproducible quarantine report and candidate files for human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d() .-]{7,}\d)(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
SPACE = re.compile(r"\s+")


def normalized(text: str) -> str:
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def fingerprint(prompt: str, answer: str) -> str:
    return hashlib.sha256((normalized(prompt) + "\n" + normalized(answer)).encode("utf-8")).hexdigest()


def shingles(text: str) -> set[str]:
    text = normalized(text).lower()
    return {text[index : index + 5] for index in range(max(0, len(text) - 4))} or {text}


def get_pair(record: dict) -> tuple[str, str]:
    if isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        return record["prompt"], record["response"]
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) == 2:
        return messages[0]["content"], messages[1]["content"]
    raise ValueError("record needs prompt/response or exactly two messages")


def rejection_reason(prompt: str, answer: str) -> str | None:
    merged = prompt + "\n" + answer
    if len(prompt) < 8 or len(answer) < 16:
        return "too_short"
    if len(prompt) > 4_000 or len(answer) > 12_000:
        return "too_long"
    if EMAIL.search(merged) or PHONE.search(merged) or CARD.search(merged):
        return "possible_personal_or_sensitive_data"
    chars = [char for char in normalized(merged) if not char.isspace()]
    if chars and Counter(chars).most_common(1)[0][1] / len(chars) > 0.45:
        return "excessive_character_repetition"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate approved-source instruction data into a review queue.")
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL; each record must include source_id and origin")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    args = parser.parse_args()

    allowlist = json.loads(args.allowlist.read_text(encoding="utf-8"))
    sources = {item["source_id"]: item for item in allowlist.get("sources", []) if item.get("approved") is True}
    if not sources:
        raise SystemExit("No approved sources in the allowlist. Add one only after confirming its licence and terms.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.out_dir / "training_candidates.jsonl"
    rejected_path = args.out_dir / "quarantine.jsonl"
    report_path = args.out_dir / "curation_report.json"
    seen_hashes: set[str] = set()
    prior_shingles: list[set[str]] = []
    counts = Counter()

    with args.input.open(encoding="utf-8") as input_handle, accepted_path.open("w", encoding="utf-8") as accepted, rejected_path.open("w", encoding="utf-8") as rejected:
        for line_number, raw in enumerate(input_handle, 1):
            counts["read"] += 1
            try:
                record = json.loads(raw)
                source_id = record.get("source_id")
                if source_id not in sources:
                    raise ValueError("source_not_approved")
                if record.get("origin") not in {"licensed_source", "public_domain", "user_owned"}:
                    raise ValueError("origin_not_allowed")
                prompt, answer = get_pair(record)
                prompt, answer = normalized(prompt), normalized(answer)
                reason = rejection_reason(prompt, answer)
                if reason:
                    raise ValueError(reason)
                digest = fingerprint(prompt, answer)
                if digest in seen_hashes:
                    raise ValueError("exact_duplicate")
                candidate_shingles = shingles(prompt + "\n" + answer)
                if any(len(candidate_shingles & old) / len(candidate_shingles | old) >= args.near_duplicate_threshold for old in prior_shingles):
                    raise ValueError("near_duplicate")
                clean = {
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                    "provenance": {"source_id": source_id, "source_url": sources[source_id]["url"], "license": sources[source_id]["license"], "license_url": sources[source_id]["license_url"]},
                    "origin": record["origin"],
                    "fingerprint": digest,
                }
                accepted.write(json.dumps(clean, ensure_ascii=False) + "\n")
                seen_hashes.add(digest)
                prior_shingles.append(candidate_shingles)
                counts["accepted"] += 1
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                reason = str(exc)
                rejected.write(json.dumps({"line": line_number, "reason": reason, "record": raw[:1_000]}, ensure_ascii=False) + "\n")
                counts[f"rejected:{reason}"] += 1
    report_path.write_text(json.dumps({"counts": counts, "allowlist": str(args.allowlist), "input": str(args.input), "automatic_training": False, "next_action": "Human review and a separate held-out evaluation are required before use."}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
