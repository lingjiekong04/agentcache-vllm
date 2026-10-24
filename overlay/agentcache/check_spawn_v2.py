"""Bounded, observable spawn check; reuse the unchanged original contract."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import check_spawn as original

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timeout_value(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 60 <= seconds <= 900:
        raise argparse.ArgumentTypeError(
            "Timeout must be finite in [60, 900] seconds"
        )
    return seconds


def child(policy, peer):
    identity = {
        "pid": os.getpid(),
        "script_sha256": digest(Path(__file__)),
        "original_script_sha256": digest(Path(original.__file__)),
    }

    def emit(kind, value):
        peer.send({"kind": kind, "value": value, **identity})

    class ResultPipe:
        def send(self, value):
            emit("result", value)

        def close(self):
            peer.close()

    try:
        emit("phase", "native_imports")
        import vllm

        emit(
            "phase",
            {"name": "native_imports_complete", "module": vllm.__file__},
        )
        # No monkeypatch: this calls the exact original CLI/pool check.
        original.child(policy, ResultPipe())
    except BaseException as error:
        try:
            emit("result", {"passed": False, "error": repr(error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        peer.close()


def wait_for_result(peer, process, seconds, on_event, clock=time.monotonic):
    deadline = clock() + seconds
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Spawn deadline exceeded; no child is killed")
        if peer.poll(min(1, remaining)):
            event = peer.recv()
            on_event(event)
            if event["kind"] == "result":
                return event["value"]
        elif not process.is_alive():
            # Recheck the pipe for a result sent immediately before exit.
            if not peer.poll(0):
                raise RuntimeError("Child exited without a result")


def validate_result(result, policy):
    queue = (
        "TwoQueuePolicy" if policy == "two_queue" else "FreeKVCacheBlockQueue"
    )
    if not result.get("passed"):
        raise ValueError(f"Child check failed: {result}")
    if (
        result.get("policy") != policy
        or Path(result["module"]).resolve()
        != ROOT / "vllm/v1/core/block_pool.py"
        or result.get("queue") != queue
        or result["stats"]["protected_capacity"]
        != (127 if policy == "two_queue" else 0)
        or result["stats"]["cold_free"] != 255
        or result["stats"]["hot_free"] != 0
    ):
        raise ValueError("Actual child policy/source/queue/capacity differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("lru", "two_queue"), required=True)
    parser.add_argument("--timeout", type=timeout_value, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "passed": False,
        "policy": args.policy,
        "timeout_seconds": args.timeout,
        "script_sha256": digest(Path(__file__)),
        "original_script_sha256": digest(Path(original.__file__)),
        "events": [],
        "scope": "Native CLI/pool spawn contract; no model serving or GPU inference",
    }
    process = None
    peer = None
    started = time.monotonic()
    # Refuse existing output before creating a child. Save failure/interruption.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
                or os.environ.get("VLLM_USE_V1") != "1"
            ):
                raise ValueError(
                    "Source agentcache/env.sh first: GPU 0 and V1 required"
                )
            manifest_path = ROOT / "agentcache/source_manifest.json"
            report["source_manifest_sha256"] = digest(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            for relative, expected in manifest["files"].items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(
                        f"Locked engineering source changed: {relative}"
                    )
            context = mp.get_context("spawn")
            peer, child_peer = context.Pipe(duplex=False)
            process = context.Process(
                target=child, args=(args.policy, child_peer)
            )
            process.start()
            child_peer.close()
            report["child_pid"] = process.pid
            print(
                f"SPAWN_STARTED pid={process.pid} timeout={args.timeout}",
                flush=True,
            )

            def record(event):
                if (
                    event["pid"] != process.pid
                    or event["script_sha256"] != report["script_sha256"]
                    or event["original_script_sha256"]
                    != report["original_script_sha256"]
                ):
                    raise ValueError("Child identity differs")
                report["events"].append(
                    {**event, "elapsed": time.monotonic() - started}
                )
                print("SPAWN_EVENT " + json.dumps(event), flush=True)

            result = wait_for_result(peer, process, args.timeout, record)
            report["result"] = result
            validate_result(result, args.policy)
            process.join(10)
            if process.exitcode != 0:
                raise RuntimeError("Child did not exit cleanly")
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            report["child_still_running"] = bool(process and process.is_alive())
            if peer is not None:
                peer.close()
            json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)
    if report["child_still_running"]:
        # multiprocessing's atexit join could otherwise wait without a bound.
        # Only the checking parent exits; the named child is not signalled.
        print(
            f"PAUSE: child PID {process.pid} is still running; inspect it manually. "
            "Do not start another check or service.",
            flush=True,
        )
        sys.stderr.flush()
        os._exit(2)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
