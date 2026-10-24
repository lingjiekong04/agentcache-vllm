"""Verify native CLI options and real spawn imports without model inference."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(policy, pipe):
    try:
        from vllm.config import CacheConfig
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils import FlexibleArgumentParser
        from vllm.v1.core import block_pool

        config = CacheConfig(
            16, 0.2, 0, "auto", enable_prefix_caching=True, gpu_eviction_policy=policy
        )
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--gpu-eviction-policy", policy, "--gpu-cache-protected-ratio", "0.5"]
        )
        assert args.gpu_eviction_policy == config.gpu_eviction_policy
        pool = block_pool.BlockPool(
            256, True, config.gpu_eviction_policy, config.gpu_cache_protected_ratio
        )
        assert Path(block_pool.__file__).resolve().is_relative_to(ROOT / "vllm")
        stats = pool.policy_stats()
        assert stats.protected_capacity == (127 if policy == "two_queue" else 0)
        pipe.send(
            {
                "passed": True,
                "policy": policy,
                "module": block_pool.__file__,
                "queue": type(pool.free_block_queue).__name__,
                "stats": vars(stats),
            }
        )
    except BaseException as error:
        pipe.send({"passed": False, "error": repr(error)})
    finally:
        pipe.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("lru", "two_queue"), required=True)
    args = p.parse_args()
    context = mp.get_context("spawn")
    parent, peer = context.Pipe(duplex=False)
    process = context.Process(target=child, args=(args.policy, peer))
    process.start()
    peer.close()
    if not parent.poll(90):
        raise TimeoutError("Spawn check timed out; inspect child, no automatic kill")
    result = parent.recv()
    process.join(10)
    print(json.dumps(result))
    if not result["passed"] or process.exitcode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
