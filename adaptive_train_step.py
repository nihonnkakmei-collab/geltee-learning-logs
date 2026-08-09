from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

def configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required local configuration: {name}")
    return Path(value)


GELTEE = configured_path("GELTEE_ROOT")
sys.path.insert(0, str(GELTEE / "scripts"))
from tokenizer import CharTokenizer
from train_model import DecoderOnlyTransformer, TrainConfig, set_seed

SOURCE_MODEL = configured_path("GELTEE_SOURCE_MODEL")
TOKENIZER = configured_path("GELTEE_TOKENIZER")
GATE_SOURCE = GELTEE / "train_v227_v171_small_vector_search.py"
TRAIN_DATA = configured_path("GELTEE_TRAIN_DATA")
HOLDOUT_DATA = configured_path("GELTEE_HOLDOUT_DATA")


def load_gate_pairs() -> dict[str, list[tuple[str, str]]]:
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8-sig"))
    wanted = {"base17", "python", "json_ascii", "safety", "jp", "explain", "logic", "math", "unseen", "gates"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                nodes.append(node)
    env: dict[str, object] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GATE_SOURCE), "exec"), {"range": range}, env)
    return env["gates"]  # type: ignore[return-value]


def sample_pairs(path: Path, seed: int, count: int, forbidden: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sample broad training or holdout examples without reading the full corpus."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    size = path.stat().st_size
    with path.open("rb") as handle:
        while len(pairs) < count:
            handle.seek(rng.randrange(size))
            handle.readline()  # discard a partial line
            for _ in range(24):
                raw = handle.readline()
                if not raw:
                    break
                try:
                    item = json.loads(raw.decode("utf-8"))
                    messages = item["messages"]
                    pair = (messages[0]["content"], messages[1]["content"])
                except (UnicodeDecodeError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                    continue
                if pair not in forbidden and pair not in seen and pair[0] and pair[1]:
                    seen.add(pair)
                    pairs.append(pair)
                    if len(pairs) == count:
                        break
    return pairs


def generate(model, tok, prompt: str, block_size: int, max_new: int = 100) -> str:
    ids = tok.encode_prompt(prompt)
    prompt_len = len(ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-block_size:]], device="cuda")
            next_id = int(torch.argmax(model(x)[0, -1]).item())
            if next_id in {tok.stoi["<|eot|>"], tok.eos_id}:
                break
            ids.append(next_id)
    return tok.decode_text(ids[prompt_len:])


def evaluate_gate(model, tok, gates, block_size: int) -> dict:
    results, score, total = {}, 0, 0
    for category, pairs in gates.items():
        failures = []
        for prompt, expected in pairs:
            actual = generate(model, tok, prompt, block_size)
            if actual == expected:
                score += 1
            else:
                failures.append({"prompt": prompt, "expected": expected, "actual": actual})
            total += 1
        results[category] = {"ok": len(pairs) - len(failures), "total": len(pairs), "fails": failures}
    return {"score": score, "total": total, "results": results}


def encode_pair(tok, prompt: str, answer: str, block_size: int):
    prompt_ids = tok.encode_prompt(prompt)
    ids = prompt_ids + tok.encode_text(answer) + [tok.stoi["<|eot|>"]]
    overflow = max(0, len(ids) - block_size)
    ids = ids[-block_size:]
    x, y = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    answer_start = max(0, len(prompt_ids) - 1 - overflow)
    weights = torch.zeros_like(y, dtype=torch.float32)
    weights[answer_start:] = 1.0
    if len(weights):
        weights[-1] = 2.0
    return x, y, weights


def answer_nll(model, tok, pairs, block_size: int) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for prompt, answer in pairs:
            x, y, weights = encode_pair(tok, prompt, answer, block_size)
            logits = model(x.unsqueeze(0).cuda())[0]
            losses = F.cross_entropy(logits, y.cuda(), reduction="none")
            values.append(float((losses * weights.cuda()).sum() / weights.sum().clamp_min(1)))
    return sum(values) / len(values)


def train_candidate(model, tok, pairs, cfg, step: int) -> dict:
    """Small, low-LR update on data that excludes every fixed gate pair."""
    rng = random.Random(910000 + step)
    rng.shuffle(pairs)
    lr = [2e-8, 3e-8, 5e-8][(step - 1) % 3]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for prompt, answer in pairs:
        x, y, weights = encode_pair(tok, prompt, answer, cfg.block_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x.unsqueeze(0).cuda())[0]
            token_loss = F.cross_entropy(logits, y.cuda(), reduction="none")
            loss = (token_loss * weights.cuda()).sum() / weights.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"lr": lr, "updates": len(losses), "mean_loss": sum(losses) / len(losses)}


def make_model(payload, cfg, tok):
    model = DecoderOnlyTransformer(tok.vocab_size, cfg, tok.pad_id)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.cuda()


def initialize_state(state_dir: Path, reset: bool) -> tuple[Path, Path, Path]:
    baseline, champion, accepted = state_dir / "baseline.pt", state_dir / "champion.pt", state_dir / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    if not baseline.exists():
        shutil.copy2(SOURCE_MODEL, baseline)
    if reset and champion.exists():
        shutil.copy2(champion, accepted / f"legacy-overfit-{int(time.time())}.pt")
    if reset or not champion.exists():
        shutil.copy2(baseline, champion)
        shutil.copy2(baseline, accepted / "step-000000.pt")
    return baseline, champion, accepted


def read_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"history": []}


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reset-to-baseline", action="store_true")
    parser.add_argument("--reset-only", action="store_true")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    _, champion_path, accepted_dir = initialize_state(args.state_dir, args.reset_to_baseline)
    manifest_path = args.state_dir / "promotion_history.json"
    if args.reset_to_baseline and manifest_path.exists():
        manifest_path.unlink()
    manifest = read_manifest(manifest_path)

    set_seed(227000 + args.step)
    tok = CharTokenizer.load(str(TOKENIZER))
    gates = load_gate_pairs()
    gate_pairs = {pair for category in gates.values() for pair in category}
    train_pairs = sample_pairs(TRAIN_DATA, 1000000 + args.step, 32, gate_pairs)
    # Fixed seed: this is a true holdout metric, not a moving target.
    holdout_pairs = sample_pairs(HOLDOUT_DATA, 2000000, 32, gate_pairs)

    payload = torch.load(champion_path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**payload["config"])
    cfg.tokenizer_file, cfg.device, cfg.dropout = str(TOKENIZER), "cuda", 0.0

    baseline_model = make_model(payload, cfg, tok)
    baseline_gate = evaluate_gate(baseline_model, tok, gates, cfg.block_size)
    baseline_holdout = answer_nll(baseline_model, tok, holdout_pairs, cfg.block_size)

    rollback = None
    if manifest["history"]:
        expected = manifest["history"][-1]
        unhealthy = (
            baseline_gate["score"] < expected["gate_score"]
            or baseline_holdout > expected["holdout_nll"] * 1.01
        )
        if unhealthy and len(manifest["history"]) > 1:
            previous = manifest["history"][-2]
            shutil.copy2(previous["path"], champion_path)
            manifest["history"].pop()
            write_manifest(manifest_path, manifest)
            rollback = {"from_step": expected["step"], "to_step": previous["step"], "reason": "champion re-evaluation regressed"}
            payload = torch.load(champion_path, map_location="cpu", weights_only=True)
            baseline_model = make_model(payload, cfg, tok)
            baseline_gate = evaluate_gate(baseline_model, tok, gates, cfg.block_size)
            baseline_holdout = answer_nll(baseline_model, tok, holdout_pairs, cfg.block_size)
    if not manifest["history"]:
        baseline_copy = accepted_dir / "step-000000.pt"
        if not baseline_copy.exists():
            shutil.copy2(champion_path, baseline_copy)
        manifest["history"] = [{"step": 0, "path": str(baseline_copy), "gate_score": baseline_gate["score"], "holdout_nll": baseline_holdout}]
        write_manifest(manifest_path, manifest)

    if args.reset_only:
        result = {
            "step": args.step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "reset_to_immutable_baseline",
            "gate_in_training": False,
            "baseline": {"gate": baseline_gate, "holdout_nll": baseline_holdout},
            "rollback": "previous champion was archived and champion reset to the original v227 checkpoint",
            "gpt1_claim": False,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"step": args.step, "decision": result["decision"], "gate": baseline_gate["score"]}, ensure_ascii=False))
        return 0

    candidate = make_model(payload, cfg, tok)
    train_info = train_candidate(candidate, tok, train_pairs, cfg, args.step)
    candidate_gate = evaluate_gate(candidate, tok, gates, cfg.block_size)
    candidate_holdout = answer_nll(candidate, tok, holdout_pairs, cfg.block_size)

    gate_safe = candidate_gate["score"] >= baseline_gate["score"]
    holdout_improved = candidate_holdout < baseline_holdout * 0.997
    promoted = gate_safe and holdout_improved
    decision = "promoted" if promoted else "rejected"
    if promoted:
        accepted_candidate = accepted_dir / f"step-{args.step:06d}.pt"
        temporary = champion_path.with_suffix(".candidate.tmp")
        torch.save({"model_state": {k: v.detach().cpu() for k, v in candidate.state_dict().items()}, "config": payload["config"], "source": "guarded_generalization_loop", "parent": str(champion_path), "step": args.step}, temporary)
        os.replace(temporary, accepted_candidate)
        shutil.copy2(accepted_candidate, champion_path)
        manifest["history"].append({"step": args.step, "path": str(accepted_candidate), "gate_score": candidate_gate["score"], "holdout_nll": candidate_holdout})
        write_manifest(manifest_path, manifest)

    result = {
        "step": args.step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_model": "geltee_v227_v171_small_vector_search_model.pt",
        "training_data": "geltee_v02_100m_mix",
        "holdout_data": "v65_broad_stable_large",
        "gate_in_training": False,
        "baseline": {"gate": baseline_gate, "holdout_nll": baseline_holdout},
        "candidate": {"gate": candidate_gate, "holdout_nll": candidate_holdout},
        "train": train_info,
        "decision": decision,
        "promotion_rule": "gate score must not decrease and holdout NLL must improve by at least 0.3%",
        "rollback": rollback or "health re-evaluation passed; immutable baseline and every accepted champion are retained locally",
        "gpt1_claim": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"step": args.step, "decision": decision, "baseline_gate": baseline_gate["score"], "candidate_gate": candidate_gate["score"], "baseline_holdout_nll": baseline_holdout, "candidate_holdout_nll": candidate_holdout}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
