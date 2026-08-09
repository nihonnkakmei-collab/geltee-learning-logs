"""Find *metadata only* for openly licensed data relevant to failed gate areas.

The output is a local discovery queue. A source never becomes training data here:
its data card, terms and intended use must first be reviewed and then entered in
the separate curation allowlist.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def failures_by_gate(status: dict) -> dict[str, int]:
    results = status.get("baseline", {}).get("gate", {}).get("results", {})
    return {name: len(result.get("fails", [])) for name, result in results.items() if result.get("fails")}


def search(endpoint: str, query: str, limit: int) -> list[dict]:
    params = urllib.parse.urlencode({"search": query, "limit": limit})
    request = urllib.request.Request(f"{endpoint}?{params}", headers={"User-Agent": "Gelqeen-research/1.0 (metadata-only)"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        return []
    return payload


def license_of(item: dict) -> str:
    card = item.get("cardData") or {}
    license_name = card.get("license") or item.get("license") or ""
    if not license_name:
        for tag in item.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_name = tag.removeprefix("license:")
                break
    return str(license_name).lower().strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a review-only data-source discovery queue from gate failures.")
    parser.add_argument("--status", type=Path, required=True, help="The local latest.json from the guarded trainer")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    status = json.loads(args.status.read_text(encoding="utf-8"))
    failures = failures_by_gate(status)
    approved_licenses = set(policy["approved_licenses"])
    candidates: list[dict] = []
    errors: list[str] = []
    for item in policy.get("seed_candidates", []):
        relevant = sorted(set(item.get("topics", [])) & set(failures))
        if relevant and item.get("license") in approved_licenses:
            candidates.append({
                "gate": ",".join(relevant),
                "failed_cases": sum(failures[name] for name in relevant),
                "query": "known Japanese instruction candidate",
                "dataset_id": item["dataset_id"],
                "url": item["url"],
                "license": item["license"],
                "last_modified": None,
                "downloads": None,
                "decision": "discovered_only_requires_allowlist_and_data_card_review"
            })
    for gate, failed_count in failures.items():
        query = policy["query_by_gate"].get(gate)
        if not query:
            continue
        try:
            for item in search(policy["catalog_endpoint"], query, policy["max_results_per_query"]):
                license_name = license_of(item)
                if license_name not in approved_licenses:
                    continue
                candidates.append({
                    "gate": gate,
                    "failed_cases": failed_count,
                    "query": query,
                    "dataset_id": item.get("id"),
                    "url": f"https://huggingface.co/datasets/{item.get('id')}",
                    "license": license_name,
                    "last_modified": item.get("lastModified"),
                    "downloads": item.get("downloads"),
                    "decision": "discovered_only_requires_allowlist_and_data_card_review"
                })
        except Exception as exc:  # Discovery must never terminate the guarded trainer.
            errors.append(f"{gate}: {type(exc).__name__}: {exc}")
    deduplicated = {item["dataset_id"]: item for item in candidates if item.get("dataset_id")}
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "weak_areas": failures,
        "candidates": list(deduplicated.values()),
        "errors": errors,
        "automatic_download": False,
        "automatic_training": False,
        "next_action": "Review each dataset card, licence and terms; add only approved sources to a local allowlist before downloading any records."
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"weak_areas": len(failures), "candidate_count": len(deduplicated), "errors": len(errors)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
