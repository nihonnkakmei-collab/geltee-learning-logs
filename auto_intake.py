"""Download a small sample from pre-approved dataset-server sources and curate it.

This is intentionally limited to source IDs pinned in research_policy.json.  It
does not accept arbitrary catalogue hits, and writes only to the local state
directory.  The guarded trainer remains the final authority on model promotion.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from curate_dataset import fingerprint, normalized, rejection_reason, shingles

SERVER = "https://datasets-server.huggingface.co"


def get_json(path: str) -> dict:
    request = urllib.request.Request(SERVER + path, headers={"User-Agent": "Gelqeen-auto-intake/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def pair_from_row(row: dict) -> tuple[str, str] | None:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for first, second in zip(conversations, conversations[1:]):
            if first.get("from") in {"human", "user"} and second.get("from") in {"gpt", "assistant"}:
                prompt, answer = first.get("value"), second.get("value")
                if isinstance(prompt, str) and isinstance(answer, str):
                    return normalized(prompt), normalized(answer)
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("output") or row.get("response") or row.get("answer")
    context = row.get("input") or row.get("context") or ""
    if not isinstance(instruction, str) or not isinstance(answer, str) or not isinstance(context, str):
        return None
    prompt = instruction if not context.strip() else f"{instruction}\n\n{context}"
    return normalized(prompt), normalized(answer)


def source_rows(dataset_id: str, limit: int) -> list[dict]:
    encoded = urllib.parse.urlencode({"dataset": dataset_id})
    splits = get_json(f"/splits?{encoded}").get("splits", [])
    if not splits:
        return []
    split = next((item for item in splits if item.get("split") == "train"), splits[0])
    rows_total = int(split.get("num_rows") or 0)
    offset = random.Random(dataset_id).randrange(max(1, rows_total - limit + 1))
    params = urllib.parse.urlencode({"dataset": dataset_id, "config": split["config"], "split": split["split"], "offset": offset, "length": limit})
    return [item.get("row", {}) for item in get_json(f"/rows?{params}").get("rows", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=64)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "training_candidates.jsonl"
    report = args.out_dir / "auto_intake_report.json"
    prior = set()
    if target.exists():
        for raw in target.read_text(encoding="utf-8").splitlines():
            try:
                prior.add(json.loads(raw)["fingerprint"])
            except (KeyError, json.JSONDecodeError):
                continue
    known_shingles: list[set[str]] = []
    accepted = 0
    errors: list[str] = []
    with target.open("a", encoding="utf-8") as handle:
        for source in policy.get("seed_candidates", []):
            try:
                for row in source_rows(source["dataset_id"], args.per_source):
                    pair = pair_from_row(row)
                    if not pair:
                        continue
                    prompt, answer = pair
                    if rejection_reason(prompt, answer):
                        continue
                    digest = fingerprint(prompt, answer)
                    candidate_shingles = shingles(prompt + "\n" + answer)
                    if digest in prior or any(len(candidate_shingles & old) / len(candidate_shingles | old) >= 0.92 for old in known_shingles):
                        continue
                    record = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "provenance": {"source_id": source["dataset_id"], "source_url": source["url"], "license": source["license"]}, "origin": "licensed_source", "fingerprint": digest}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    prior.add(digest)
                    known_shingles.append(candidate_shingles)
                    accepted += 1
            except Exception as exc:
                errors.append(f"{source.get('dataset_id')}: {type(exc).__name__}: {exc}")
    report.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accepted_new": accepted, "errors": errors, "automatic_training": "guarded_candidate_updates_only"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_new": accepted, "errors": len(errors)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
