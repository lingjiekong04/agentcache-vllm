"""GPU lifecycle acceptance: concurrency, stream abort and pressure preemption.

Not a performance benchmark. A lack of observed preemption is an unmet gate,
not a success. No cache reset endpoint or process termination is used.
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

from demo import BASE, evidence, invoke, metrics, read, safe_diagnostic
from workloads import PROFILES


def idle(blocks, timeout=40):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        snapshot = metrics()
        free = sum(
            snapshot[f"vllm:gpu_cache_policy_{name}_blocks"]
            for name in ("cold_free", "hot_free")
        )
        if (
            snapshot["vllm:num_requests_running"] == 0
            and snapshot["vllm:num_requests_waiting"] == 0
            and free == blocks - 1
        ):
            return snapshot
        time.sleep(0.2)
    raise TimeoutError("Engine did not return to idle/free capacity; preserve service")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("lru", "two_queue"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {
        "complete": False,
        "policy": args.policy,
        "concurrent": [],
        "preemption": [],
        "scope": "GPU lifecycle, not throughput comparison",
    }
    with args.output.open("x", encoding="utf-8") as out:
        try:
            manifest = read(args.manifest)
            profile = manifest["profile"]
            config = PROFILES[profile]
            report["evidence"] = evidence(args.server_log, profile, args.policy, 4)
            initial = metrics()
            if initial["vllm:gpu_prefix_cache_queries_total"] != 0:
                raise ValueError("Lifecycle acceptance requires fresh service")
            base = manifest["scenarios"]["scan_return"][5]
            report["warmup"] = invoke(base)
            idle(config["blocks"])
            records = [
                {
                    **base,
                    "id": f"concurrent/{i}",
                    "request": {**base["request"], "max_tokens": 64, "min_tokens": 64},
                }
                for i in range(16)
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for future in concurrent.futures.as_completed(
                    [pool.submit(invoke, r) for r in records]
                ):
                    report["concurrent"].append(future.result())
            report["after_concurrent"] = idle(config["blocks"])

            body = {
                **base["request"],
                "max_tokens": 512,
                "min_tokens": 512,
                "stream": True,
            }
            request = urllib.request.Request(
                BASE + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                first = response.readline().decode()
                if not first.startswith("data: ") or "[DONE]" in first:
                    raise ValueError("No active streaming request to cancel")
                chunk = json.loads(first.removeprefix("data: "))
                if chunk["choices"][0]["finish_reason"] is not None:
                    raise ValueError("Stream finished before cancellation")
            report["cancel"] = {
                "request": body,
                "first_sse_line": first,
                "after_disconnect": idle(config["blocks"]),
            }
            # Independent prompts start together and grow beyond the bounded pool.
            generation = 768 if profile == "quick" else 1536
            before_pressure = metrics()
            records = []
            for i in range(4):
                record = manifest["scenarios"]["cold_only"][5 + i]
                records.append(
                    {
                        **record,
                        "id": f"preempt/{i}",
                        "request": {
                            **record["request"],
                            "max_tokens": generation,
                            "min_tokens": generation,
                        },
                    }
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for future in concurrent.futures.as_completed(
                    [pool.submit(invoke, r) for r in records]
                ):
                    report["preemption"].append(future.result())
            final = idle(config["blocks"])
            delta = (
                final["vllm:num_preemptions_total"]
                - before_pressure["vllm:num_preemptions_total"]
            )
            report["preemptions_observed"] = delta
            report["after_pressure"] = final
            if delta <= 0:
                raise ValueError("No preemption observed; lifecycle gate NOT validated")
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
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
    print(f"lifecycle_passed=true saved={args.output}")


if __name__ == "__main__":
    main()
