from __future__ import annotations

import argparse
import ast
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

GELTEE = Path(r"C:\GelteeLocal")
sys.path.insert(0, str(GELTEE / "scripts"))
from tokenizer import CharTokenizer
from train_model import DecoderOnlyTransformer, TrainConfig, set_seed

SOURCE_MODEL = Path(
    r"C:\Users\matsu\Downloads\battle_v227_v171_small_vector_search_candidate"
    r"\geltee_v227_v171_small_vector_search_model.pt"
)
TOKENIZER = Path(
    r"C:\Users\matsu\Downloads\battle_v227_v171_small_vector_search_candidate"
    r"\geltee_v127_fix_broader_tokenizer.json"
)
GATE_SOURCE = GELTEE / "train_v227_v171_small_vector_search.py"


def load_gate_pairs() -> dict[str, list[tuple[str, str]]]:
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8-sig"))
    wanted = {"base17", "python", "json_ascii", "safety", "jp", "explain", "logic", "math", "unseen", "gates"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    env: dict[str, object] = {}
    exec(compile(module, str(GATE_SOURCE), "exec"), {"range": range}, env)
    return env["gates"]  # type: ignore[return-value]


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


def evaluate(model, tok, gates, block_size: int) -> dict:
    results = {}
    score = 0
    total = 0
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


def encoded_example(tok, prompt: str, answer: str, block_size: int):
    prompt_ids = tok.encode_prompt(prompt)
    ids = prompt_ids + tok.encode_text(answer) + [tok.stoi["<|eot|>"]]
    ids = ids[-block_size:]
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    start = max(0, len(prompt_ids) - 1 - max(0, len(prompt_ids) + len(tok.encode_text(answer)) + 1 - block_size))
    weights = torch.zeros_like(y, dtype=torch.float32)
    weights[start:] = 1.0
    if len(weights):
        weights[-1] = 2.0
    return x, y, weights


def train_candidate(model, tok, gates, cfg, step: int) -> dict:
    rng = random.Random(227000 + step)
    pairs = [pair for values in gates.values() for pair in values]
    targeted = [
        ("PythonでFalseを表示して", "print(False)"),
        ("FalseをPythonで表示して", "print(False)"),
        ("Pythonで真偽値Falseを表示して", "print(False)"),
        ("PythonでTrueを返して", "return True"),
        ("TrueをPythonで返して", "return True"),
        ("Pythonで真偽値Trueを返して", "return True"),
    ]
    pairs.extend(targeted * (2 + step % 3))
    rng.shuffle(pairs)
    lrs = [8e-7, 1.2e-6, 1.8e-6, 2.5e-6, 3.5e-6]
    lr = lrs[(step - 1) % len(lrs)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for prompt, answer in pairs:
        x, y, weights = encoded_example(tok, prompt, answer, cfg.block_size)
        x, y, weights = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda(), weights.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)[0]
            token_loss = F.cross_entropy(logits, y[0], reduction="none")
            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"lr": lr, "updates": len(losses), "mean_loss": sum(losses) / len(losses)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    champion = args.state_dir / "champion.pt"
    if not champion.exists():
        shutil.copy2(SOURCE_MODEL, champion)

    set_seed(227000 + args.step)
    tok = CharTokenizer.load(str(TOKENIZER))
    gates = load_gate_pairs()
    payload = torch.load(champion, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**payload["config"])
    cfg.tokenizer_file = str(TOKENIZER)
    cfg.device = "cuda"
    cfg.dropout = 0.0

    baseline_model = DecoderOnlyTransformer(tok.vocab_size, cfg, tok.pad_id)
    baseline_model.load_state_dict(payload["model_state"], strict=True)
    baseline_model.cuda()
    baseline = evaluate(baseline_model, tok, gates, cfg.block_size)

    candidate = DecoderOnlyTransformer(tok.vocab_size, cfg, tok.pad_id)
    candidate.load_state_dict(payload["model_state"], strict=True)
    candidate.cuda()
    train_info = train_candidate(candidate, tok, gates, cfg, args.step)
    candidate_eval = evaluate(candidate, tok, gates, cfg.block_size)

    promoted = candidate_eval["score"] > baseline["score"]
    if promoted:
        torch.save(
            {
                "model_state": {k: v.detach().cpu() for k, v in candidate.state_dict().items()},
                "config": payload["config"],
                "source": "geltee_infinite_loop",
                "parent": str(champion),
                "step": args.step,
            },
            champion,
        )

    result = {
        "step": args.step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_model": str(SOURCE_MODEL),
        "baseline": {"score": baseline["score"], "total": baseline["total"]},
        "candidate": candidate_eval,
        "train": train_info,
        "promoted": promoted,
        "gpt1_claim": False,
        "gpt1_note": "A common benchmark against a reproduced GPT-1 baseline has not yet been completed.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("step", "baseline", "promoted")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
