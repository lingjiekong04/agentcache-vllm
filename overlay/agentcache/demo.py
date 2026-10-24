"""Unified manifest creation, serial HTTP collection and honest paired reports."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from workloads import PROFILES, SCENARIOS, digest, generate

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8002"
GAUGES = ("cold_free", "hot_free", "protected", "protected_capacity")
COUNTERS = ("promotions", "demotions", "cold_evictions", "hot_evictions")


def save(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)


def safe_diagnostic(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_value": repr(value)}
    if isinstance(value, dict):
        return {k: safe_diagnostic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_diagnostic(v) for v in value]
    return value


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError(f"Invalid number: {s}")
        ),
    )


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=15) as response:
        data = response.read().decode()
    result = {}
    for line in data.splitlines():
        if line.startswith("vllm:"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name.startswith("vllm:gpu_cache_policy_") or name in (
                "vllm:gpu_prefix_cache_queries_total",
                "vllm:gpu_prefix_cache_hits_total",
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:num_preemptions_total",
            ):
                if name in result:
                    raise ValueError("Multiple engine series are out of scope")
                result[name] = float(line.rsplit(" ", 1)[-1])
    required = [f"vllm:gpu_cache_policy_{f}_blocks" for f in GAUGES]
    required += [f"vllm:gpu_cache_policy_{f}_total" for f in COUNTERS]
    required += [
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total",
    ]
    if any(
        key not in result or not math.isfinite(result[key]) for key in required
    ):
        raise ValueError("Native policy metrics missing or nonfinite")
    return result


def invoke(record):
    data = json.dumps(record["request"]).encode()
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    elapsed = time.perf_counter() - start
    # JSON parsing and validation are deliberately outside the timing interval.
    try:
        result = json.loads(raw)
    except Exception as error:
        error.failed_response = {
            **record,
            "raw_response": raw.decode(errors="replace"),
        }
        raise
    row = {**record, "response": result, "latency_ms": elapsed * 1000}
    try:
        token_ids = validate_response(record, result)
    except Exception as error:
        error.failed_response = row
        raise
    return {**row, "token_ids": token_ids}


def validate_response(record, result):
    choice = result["choices"][0]
    usage = result["usage"]
    count = record["request"]["max_tokens"]
    if (
        usage["completion_tokens"] != count
        or choice["finish_reason"] != "length"
        or usage["prompt_tokens"] != len(record["request"]["prompt"])
        or usage["total_tokens"] != usage["prompt_tokens"] + count
    ):
        raise ValueError("Response length/usage/finish reason differs")
    lp = choice["logprobs"]
    token_ids = [int(token.removeprefix("token_id:")) for token in lp["tokens"]]
    scores = list(lp["token_logprobs"])
    scores += [
        v for candidates in lp["top_logprobs"] for v in candidates.values()
    ]
    if len(token_ids) != count or not all(math.isfinite(v) for v in scores):
        raise ValueError("Invalid token IDs/logprobs")
    return token_ids


def evidence(log, profile, policy, concurrency):
    raw = log.read_bytes()
    records = []
    for line in raw.decode().splitlines():
        marker = "AGENTCACHE_NATIVE_ENGINE "
        if marker in line:
            records.append(json.loads(line.split(marker, 1)[1]))
    if len(records) != 1:
        raise ValueError(
            "Need exactly one native EngineCore launch in fresh log"
        )
    if "disable_log_requests=True" not in raw.decode():
        raise ValueError("Frontend full request logging must be disabled")
    record = records[0]
    if Path(record["source_root"]).resolve() != (ROOT / "vllm").resolve():
        raise ValueError("Actual EngineCore did not import engineering source")
    required_sources = {
        "config.py",
        "engine/arg_utils.py",
        "v1/core/block_pool.py",
        "v1/core/gpu_eviction_policy.py",
        "v1/core/kv_cache_manager.py",
        "v1/core/sched/scheduler.py",
        "v1/metrics/stats.py",
        "v1/metrics/loggers.py",
        "v1/core/agentcache_identity.py",
        "v1/engine/core.py",
    }
    if set(record["source_sha256"]) != required_sources:
        raise ValueError("Incomplete EngineCore source identity")
    cfg, expected = record["config"], PROFILES[profile]
    for key, value in {
        "model": expected["model"],
        "dtype": "torch.bfloat16",
        "max_model_len": expected["context"],
        "blocks": expected["blocks"],
        "block_size": 16,
        "protected_ratio": 0.5,
        "apc": True,
        "seed": 0,
        "max_num_seqs": concurrency,
        "gpu_memory_utilization": 0.2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }.items():
        if cfg.get(key) != value:
            raise ValueError(f"Actual EngineCore config differs: {key}")
    if record["policy"] != policy:
        raise ValueError("Actual EngineCore policy differs")
    queue = (
        "TwoQueuePolicy" if policy == "two_queue" else "FreeKVCacheBlockQueue"
    )
    if not record["queue_type"].endswith("." + queue):
        raise ValueError("Actual queue type differs")
    capacity = (
        max(1, int((expected["blocks"] - 1) * 0.5))
        if policy == "two_queue"
        else 0
    )
    if record["policy_stats"]["protected_capacity"] != capacity:
        raise ValueError("Actual protected capacity differs")
    for file, value in record["source_sha256"].items():
        if (
            hashlib.sha256((ROOT / "vllm" / file).read_bytes()).hexdigest()
            != value
        ):
            raise ValueError(f"EngineCore source hash differs: {file}")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if (
        record["visible_devices"] != "0"
        or gpu.split(",")[0].strip() != record["gpu_uuid"]
    ):
        raise ValueError("Actual EngineCore did not use expected GPU 0")
    return {
        "engine": record,
        "gpu": gpu,
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "log_snapshot": raw.decode(),
    }


def summary(records):
    result = {}
    for phase in dict.fromkeys(r["phase"] for r in records):
        rows = [r for r in records if r["phase"] == phase]
        times = sorted(r["latency_ms"] for r in rows)
        if not all(math.isfinite(x) and x > 0 for x in times):
            raise ValueError("Invalid HTTP latency")
        queries = sum(r["cache_queries"] for r in rows)
        hits = sum(r["cache_hits"] for r in rows)
        result[phase] = {
            "requests": len(rows),
            "mean_ms": statistics.mean(times),
            "p50_ms": statistics.median(times),
            "p95_ms": times[max(0, math.ceil(0.95 * len(times)) - 1)],
            "http_service_requests_s": len(rows) / (sum(times) / 1000),
            "cache_queries": queries,
            "cache_hits": hits,
            "hit_ratio": hits / queries if queries else None,
        }
    return result


def run(args):
    report = {
        "complete": False,
        "policy": args.policy,
        "scenario": args.scenario,
        "records": [],
        "performance_conclusions_allowed": False,
    }
    # Reserve output before any network requests; failed evidence is retained.
    with args.output.open("x", encoding="utf-8") as out:
        try:
            manifest = read(args.manifest)
            if manifest["input_sha256"] != digest(manifest["scenarios"]):
                raise ValueError("Manifest input hash differs")
            report.update(
                profile=manifest["profile"],
                input_sha256=manifest["input_sha256"],
                evidence=evidence(
                    args.server_log, manifest["profile"], args.policy, 1
                ),
            )
            records = manifest["scenarios"][args.scenario]
            report["input_records"] = records
            report["sequence_sha256"] = digest(records)
            initial = metrics()
            if initial["vllm:gpu_prefix_cache_queries_total"] != 0:
                raise ValueError(
                    "Expected fresh service with no generation traffic"
                )
            for index, record in enumerate(records):
                before = metrics()
                row = invoke(record)
                report["records"].append(row)
                after = metrics()
                row.update(
                    metrics_before=before,
                    metrics_after=after,
                    cache_queries=after["vllm:gpu_prefix_cache_queries_total"]
                    - before["vllm:gpu_prefix_cache_queries_total"],
                    cache_hits=after["vllm:gpu_prefix_cache_hits_total"]
                    - before["vllm:gpu_prefix_cache_hits_total"],
                )
                if (
                    row["cache_queries"] < 0
                    or not 0 <= row["cache_hits"] <= row["cache_queries"]
                ):
                    raise ValueError("Invalid cumulative metric deltas")
                print(
                    f"{index + 1}/{len(records)} {record['id']} "
                    f"ms={row['latency_ms']:.3f} "
                    f"hits={row['cache_hits']}/{row['cache_queries']}",
                    flush=True,
                )
            report["summary"] = summary(report["records"])
            report["complete"] = True
        except BaseException as error:
            report["error"] = repr(error)
            if hasattr(error, "failed_response"):
                report["failed_response"] = error.failed_response
            raise
        finally:
            json.dump(
                safe_diagnostic(report),
                out,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
    print(f"complete=true requests={len(records)} saved={args.output}")


def compare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    left, right = read(args.lru), read(args.two_queue)
    for report, policy in ((left, "lru"), (right, "two_queue")):
        if not report["complete"] or report["policy"] != policy:
            raise ValueError("Incomplete or wrong-policy input")
        count = {
            "agent_sessions": 17,
            "scan_return": 75,
            "working_set_shift": 30,
            "cold_only": 45,
        }[report["scenario"]]
        actual_inputs = [
            {k: row[k] for k in ("id", "phase", "request")}
            for row in report["records"]
        ]
        if (
            len(actual_inputs) != count
            or actual_inputs != report["input_records"]
            or digest(actual_inputs) != report["sequence_sha256"]
        ):
            raise ValueError("Stored request sequence differs")
        for row in report["records"]:
            if validate_response(row, row["response"]) != row["token_ids"]:
                raise ValueError("Stored token IDs differ from response")
            q = row["cache_queries"]
            h = row["cache_hits"]
            if not math.isfinite(q) or not math.isfinite(h) or not 0 <= h <= q:
                raise ValueError("Invalid cached metric delta")
    for field in ("profile", "scenario", "input_sha256"):
        if left[field] != right[field]:
            raise ValueError(f"Unpaired {field}")
    for field in ("gpu",):
        if left["evidence"][field] != right["evidence"][field]:
            raise ValueError("GPU/driver changed")
    for field in ("source_sha256", "config"):
        if (
            left["evidence"]["engine"][field]
            != right["evidence"]["engine"][field]
        ):
            raise ValueError(f"Engine evidence differs: {field}")
    if (
        left["evidence"]["engine"]["run_id"]
        == right["evidence"]["engine"]["run_id"]
    ):
        raise ValueError("Comparison requires independent service runs")
    if len(left["records"]) != len(right["records"]):
        raise ValueError("Different request counts")
    mismatches = []
    for a, b in zip(left["records"], right["records"]):
        if a["id"] != b["id"] or a["request"] != b["request"]:
            raise ValueError("Request pairing differs")
        ca, cb = a["response"]["choices"][0], b["response"]["choices"][0]
        fields = [
            name
            for name, x, y in (
                ("tokens", a["token_ids"], b["token_ids"]),
                ("text", ca["text"], cb["text"]),
                ("usage", a["response"]["usage"], b["response"]["usage"]),
                ("finish_reason", ca["finish_reason"], cb["finish_reason"]),
            )
            if x != y
        ]
        if fields:
            mismatches.append(
                {"id": a["id"], "fields": fields, "lru": a, "two_queue": b}
            )
    summaries = {
        "lru": summary(left["records"]),
        "two_queue": summary(right["records"]),
    }
    for report in (left, right):
        if summary(report["records"]) != report["summary"]:
            raise ValueError("Stored summary does not recompute")
    pressure_valid = None
    if left["scenario"] == "scan_return":
        pressure_valid = all(
            (
                r["cache_hits"] == 0
                if policy == "lru"
                else r["cache_queries"] > 0
                and r["cache_hits"] / r["cache_queries"] >= 0.9
            )
            for report, policy in ((left, "lru"), (right, "two_queue"))
            for r in report["records"]
            if r["phase"] == "after"
        )
        pressure_valid = pressure_valid and all(
            r["cache_queries"] > 0
            and r["cache_hits"] / r["cache_queries"] >= 0.9
            for report in (left, right)
            for r in report["records"]
            if r["phase"] == "before"
        )
    result = {
        "paired_requests": len(left["records"]),
        "output_mismatches": mismatches,
        "pressure_valid": pressure_valid,
        "summary": summaries,
        "review_required": bool(mismatches),
        "performance_conclusions_allowed": False,
        "scope": "single paired run; not production speedup or TTFT",
    }
    save(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "output_mismatches"},
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--profile", choices=PROFILES, required=True)
    build.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--policy", choices=("lru", "two_queue"), required=True
    )
    run_parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--server-log", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    for name in ("lru", "two-queue", "output"):
        compare_parser.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.command == "build":
        if args.output.exists():
            raise FileExistsError(args.output)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            PROFILES[args.profile]["model"], local_files_only=True
        )
        manifest = generate(
            args.profile,
            lambda s: tokenizer.encode(s, add_special_tokens=False),
        )
        manifest["tokenizer_class"] = type(tokenizer).__name__
        save(args.output, manifest)
        print(f"saved={args.output} input_sha256={manifest['input_sha256']}")
    elif args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
