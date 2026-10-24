"""Fixed inference trajectories; tool outputs are recordings, never executed."""

import hashlib
import json

PROFILES = {
    "quick": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "context": 2048,
        "blocks": 256,
        "hot_tokens": 704,
        "cold_tokens": 832,
    },
    "demo": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "context": 8192,
        "blocks": 1024,
        "hot_tokens": 3072,
        "cold_tokens": 4096,
    },
}
SCENARIOS = ("agent_sessions", "scan_return", "working_set_shift", "cold_only")
CONTEXT = (
    "The project is a read-only inventory assistant. Tool results are fixed "
    "recordings. Warehouse A has 20 routers and warehouse B has 12 switches. "
    "Never execute shell commands or make purchases. Explain the recorded "
    "inventory and identify missing information. Keep the project context unchanged. "
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate(profile, encode):
    """Generate once using a cached tokenizer, then reuse exact token inputs."""
    cfg = PROFILES[profile]

    def fill(header, size):
        tokens = encode(header + CONTEXT * (size // 8 + 2))
        if len(tokens) < size:
            raise ValueError("Tokenizer produced insufficient context tokens")
        return tokens[:size]

    cold_first = set()

    def cold(label):
        nonce = hashlib.sha256(label.encode()).hexdigest()
        text = f"{nonce} Independent request {label}. "
        prompt = fill(text, cfg["cold_tokens"])
        key = tuple(prompt[:16])
        if key in cold_first:
            raise ValueError("Cold inputs share their first block")
        cold_first.add(key)
        return prompt

    def hot(label):
        return fill(f"Shared project {label}. ", cfg["hot_tokens"])

    def request(case_id, phase, prompt, tokens=16):
        if len(prompt) + tokens > cfg["context"]:
            raise ValueError("Request exceeds context")
        return {
            "id": case_id,
            "phase": phase,
            "request": {
                "model": cfg["model"],
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": tokens,
                "min_tokens": tokens,
                "ignore_eos": True,
                "seed": 0,
                "logprobs": 5,
                "return_tokens_as_token_ids": True,
            },
        }

    scenarios = {}
    common = hot("inventory")
    agent = []
    for turn in range(3):
        for session in range(4):
            transcript = (
                f"\nSession {session}, turn {turn}. User: summarize available stock.\n"
                "Recorded tool inventory.lookup: {routers:20, switches:12}.\n"
                "Assistant (recorded previous turn): inventory is read-only.\n"
                "Assistant:"
            )
            agent.append(
                request(
                    f"agent/{session}/{turn}",
                    "multi_turn",
                    common + encode(transcript),
                    32,
                )
            )
    scenarios["agent_sessions"] = agent

    scan = []
    for round_id in range(5):
        prefix = hot(f"scan-hot-{round_id}")
        for i in range(4):
            scan.append(request(f"scan/{round_id}/warm/{i}", "warm", prefix, 1))
        scan.append(request(f"scan/{round_id}/before", "before", prefix, 1))
        for i in range(8):
            scan.append(
                request(
                    f"scan/{round_id}/cold/{i}",
                    "cold_scan",
                    cold(f"scan-{round_id}-{i}"),
                    1,
                )
            )
        scan.append(request(f"scan/{round_id}/after", "after", prefix, 1))
    if 8 * (cfg["cold_tokens"] // 16) <= cfg["blocks"] - 1:
        raise ValueError("Cold scan does not exceed usable cache capacity")
    scenarios["scan_return"] = scan

    shift = []
    for epoch in range(2):
        for group in range(4):
            prefix = hot(f"epoch-{epoch}-set-{group}")
            for i in range(3):
                shift.append(
                    request(
                        f"shift/{epoch}/{group}/{i}",
                        "old_set" if epoch == 0 else "new_set",
                        prefix,
                        1,
                    )
                )
    shift.append(request("shift/old_return", "old_return", hot("epoch-0-set-0"), 1))
    scenarios["working_set_shift"] = shift
    scenarios["cold_only"] = [
        request(f"cold/{i}", "cold_only", cold(f"cold-{i}"), 1) for i in range(40)
    ]
    warm = [
        request(f"engine/warm/{i}", "engine_warm", cold(f"warm-{i}"), 1)
        for i in range(5)
    ]
    for records in scenarios.values():
        records[:0] = warm
    manifest = {
        "schema": 1,
        "profile": profile,
        "config": cfg,
        "scope": "fixed recorded-tool inference trajectories",
        "scenarios": scenarios,
    }
    manifest["input_sha256"] = digest(scenarios)
    return manifest
