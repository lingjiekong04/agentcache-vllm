"""CPU contract checks. --extracted is local metadata-only, not native pytest."""

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import math
import random
import sys
import tarfile
import types
import unittest
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Protocol

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "99078eb047a302bcf553aa3c84230631cd4b1288927c326a8be9c714b08faffb"


def load_ast(module, path, names):
    tree = ast.parse(path.read_bytes())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing definitions: {path}")
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *nodes,
    ]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)


def modules(extracted):
    if not extracted:
        from vllm.v1.core import block_pool, gpu_eviction_policy, kv_cache_utils
        from vllm.config import CacheConfig

        return (block_pool.BlockPool, kv_cache_utils, gpu_eviction_policy, CacheConfig)
    m = types.ModuleType("agentcache_extracted")
    sys.modules[m.__name__] = m
    m.__dict__.update(
        dataclass=dataclass,
        field=field,
        defaultdict=defaultdict,
        OrderedDict=OrderedDict,
        Iterable=Iterable,
        Sequence=Sequence,
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        Protocol=Protocol,
        math=math,
        logger=logging.getLogger(m.__name__),
    )
    load_ast(
        m,
        ROOT / "vllm/v1/core/kv_cache_utils.py",
        ("KVCacheBlock", "BlockHashType", "FreeKVCacheBlockQueue"),
    )
    load_ast(m, ROOT / "vllm/v1/metrics/stats.py", ("GPUCachePolicyStats",))
    load_ast(
        m,
        ROOT / "vllm/v1/core/gpu_eviction_policy.py",
        ("validate_policy", "GPUCachePolicy", "TwoQueuePolicy"),
    )
    load_ast(m, ROOT / "vllm/v1/core/block_pool.py", ("BlockPool",))
    cfg = types.ModuleType("agentcache_config_extracted")
    sys.modules[cfg.__name__] = cfg
    cfg.__dict__.update(
        GiB_bytes=1024**3, logger=m.logger, envs=types.SimpleNamespace(VLLM_USE_V1=True)
    )
    load_ast(cfg, ROOT / "vllm/config.py", ("CacheConfig",))
    return m.BlockPool, m, m, cfg.CacheConfig


def stock_class(archive, utils):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Official baseline archive SHA256 differs")
    with tarfile.open(archive) as tar:
        data = tar.extractfile("vllm-0.8.3/vllm/v1/core/block_pool.py").read()
    m = types.ModuleType("stock_pool")
    m.__dict__.update(vars(utils))
    m.__dict__.update(
        defaultdict=defaultdict, Iterable=Iterable, logger=logging.getLogger("stock")
    )
    tree = ast.parse(data)
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BlockPool"
        ),
    ]
    exec(compile(ast.fix_missing_locations(tree), "stock-v083", "exec"), m.__dict__)
    return m.BlockPool


def frozen_class(reference, stock, utils):
    expected = {
        "policy.py": "c17f572d74c83695b8ff41700672cec5504d04f9700370f60832058a5e792b7f",
        "policy_v2.py": (
            "ad2573d72a08da63f0ed25ee45424a5cd228422905a8ef993cdfa6495f23f499"
        ),
    }
    for file, digest in expected.items():
        if hashlib.sha256((reference / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen reference changed: {file}")
    for name in ("policy", "policy_v2"):
        spec = importlib.util.spec_from_file_location(name, reference / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.make_pool_class(
        stock, utils.FreeKVCacheBlockQueue, utils.KVCacheBlock, 0.5
    )


def state(pool):
    queue = pool.free_block_queue
    result = {
        "free": [b.block_id for b in queue.get_all_free_blocks()],
        "blocks": [
            (
                b.block_id,
                b.ref_cnt,
                b.block_hash,
                getattr(b.prev_free_block, "block_id", None),
                getattr(b.next_free_block, "block_id", None),
            )
            for b in pool.blocks
        ],
        "hashes": [(k, list(v)) for k, v in pool.cached_block_hash_to_block.items()],
    }
    if hasattr(queue, "protected"):
        result.update(
            protected=list(queue.protected),
            cold=[b.block_id for b in queue.cold.get_all_free_blocks()],
            hot=[b.block_id for b in queue.hot.get_all_free_blocks()],
        )
    return result


def invariants(pool):
    free = pool.free_block_queue.get_all_free_blocks()
    ids = [b.block_id for b in free]
    assert len(ids) == len(set(ids)) == pool.get_num_free_blocks()
    assert set(ids) == {
        b.block_id for b in pool.blocks if b.ref_cnt == 0 and b is not pool.null_block
    }
    assert all(b.ref_cnt >= 0 for b in pool.blocks)
    for b in free:
        assert b.ref_cnt == 0 and b is not pool.null_block
    for b in pool.blocks:
        if b.block_hash is not None:
            assert pool.cached_block_hash_to_block[b.block_hash][b.block_id] is b
    q = pool.free_block_queue
    if hasattr(q, "protected"):
        assert len(q.protected) <= q.protected_capacity
        assert set(q.members) == set(ids)
        assert all(b.block_hash is not None for b in q.protected.values())
        for b in free:
            assert (b.block_id in q.protected) == (q.members[b.block_id] is q.hot)


def build_tests(native, utils, policy, config, stock, frozen):
    def cache(pool, ids, keys=None):
        if keys is None:
            keys = ids
        hashes = [utils.BlockHashType(k, (k,) * 16) for k in keys]
        pool.cache_full_blocks(
            None, [pool.blocks[i] for i in ids], hashes, 0, len(ids), 16, hash
        )

    class Contracts(unittest.TestCase):
        compared_states = 0

        def test_config_validation_and_lru_defaults(self):
            c = config(16, 0.2, 0, "auto")
            self.assertEqual(c.gpu_eviction_policy, "lru")
            for ratio in (0, -1, 1.1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    config(
                        16,
                        0.2,
                        0,
                        "auto",
                        enable_prefix_caching=True,
                        gpu_eviction_policy="two_queue",
                        gpu_cache_protected_ratio=ratio,
                    )
            with self.assertRaises(ValueError):
                config(16, 0.2, 0, "auto", gpu_eviction_policy="two_queue")
            with self.assertRaises(ValueError):
                config(
                    16,
                    0.2,
                    0,
                    "auto",
                    enable_prefix_caching=True,
                    sliding_window=128,
                    gpu_eviction_policy="two_queue",
                )
            with self.assertRaises(ValueError):
                native(8, True, "unknown")
            self.assertIsNone(native(8, False).eviction_policy)

        def test_config_rejects_unsupported_engine_combinations(self):
            def valid():
                return types.SimpleNamespace(
                    device_config=types.SimpleNamespace(device_type="cuda"),
                    parallel_config=types.SimpleNamespace(
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        data_parallel_size=1,
                    ),
                    model_config=types.SimpleNamespace(
                        is_multimodal_model=False,
                        is_encoder_decoder=False,
                        use_mla=False,
                        runner_type="generate",
                    ),
                    lora_config=None,
                    prompt_adapter_config=None,
                    speculative_config=None,
                    kv_transfer_config=None,
                )

            cfg = config(
                16,
                0.2,
                0,
                "auto",
                enable_prefix_caching=True,
                gpu_eviction_policy="two_queue",
            )
            # Set only the extracted environment; native tests keep real V1 env.
            namespace = sys.modules[config.__module__]
            if config.__module__ == "agentcache_config_extracted":
                namespace.envs.VLLM_USE_V1 = False
                with self.assertRaises(ValueError):
                    cfg.verify_gpu_policy_with_vllm_config(valid())
                namespace.envs.VLLM_USE_V1 = True
            cfg.verify_gpu_policy_with_vllm_config(valid())
            for option, message in (("device_config", "DeviceConfig"),
                                    ("parallel_config", "single GPU")):
                value = valid()
                setattr(value, option, None)
                with self.assertRaisesRegex(ValueError, message):
                    cfg.verify_gpu_policy_with_vllm_config(value)
            for option in (
                "lora_config",
                "prompt_adapter_config",
                "speculative_config",
                "kv_transfer_config",
            ):
                value = valid()
                setattr(value, option, object())
                with self.assertRaises(ValueError):
                    cfg.verify_gpu_policy_with_vllm_config(value)
            for option in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            ):
                value = valid()
                setattr(value.parallel_config, option, 2)
                with self.assertRaises(ValueError):
                    cfg.verify_gpu_policy_with_vllm_config(value)
            for option in ("is_multimodal_model", "is_encoder_decoder", "use_mla"):
                value = valid()
                setattr(value.model_config, option, True)
                with self.assertRaises(ValueError):
                    cfg.verify_gpu_policy_with_vllm_config(value)
            value = valid()
            value.device_config.device_type = "cpu"
            with self.assertRaises(ValueError):
                cfg.verify_gpu_policy_with_vllm_config(value)

        def trace(self, left, right, seed):
            rng = random.Random(seed)
            for _ in range(2500):
                active = [b.block_id for b in left.blocks[1:] if b.ref_cnt > 0]
                cached = [
                    b.block_id for b in left.blocks[1:] if b.block_hash is not None
                ]
                operation = rng.randrange(6)
                if operation == 0:
                    count = rng.randrange(1, 5)
                    for pool in (left, right):
                        if count <= pool.get_num_free_blocks():
                            ids = [b.block_id for b in pool.get_new_blocks(count)]
                            cache(pool, ids, [rng_key for rng_key in ids])
                        else:
                            before = state(pool)
                            with self.assertRaises(ValueError):
                                pool.get_new_blocks(count)
                            self.assertEqual(state(pool), before)
                elif operation == 1 and active:
                    ids = rng.sample(active, min(len(active), 4))
                    for pool in (left, right):
                        pool.free_blocks(pool.blocks[i] for i in reversed(ids))
                elif operation in (2, 3) and cached:
                    ids = rng.sample(cached, min(len(cached), 3))
                    for pool in (left, right):
                        pool.touch([pool.blocks[i] for i in ids])
                elif operation == 4 and not active:
                    self.assertEqual(
                        left.reset_prefix_cache(), right.reset_prefix_cache()
                    )
                elif operation == 5:
                    for pool in (left, right):
                        if hasattr(pool.free_block_queue, "protected") and cached:
                            pool.free_block_queue.promote(pool.blocks[cached[0]])
                for pool in (left, right):
                    invariants(pool)
                self.assertEqual(state(left), state(right))
                self.__class__.compared_states += 1

        def test_default_lru_matches_official_v083(self):
            for seed in range(3):
                self.trace(native(33, True), stock(33, True), seed)

        def test_two_queue_matches_frozen_v2(self):
            for seed in range(3):
                self.trace(native(33, True, "two_queue"), frozen(33, True), seed)

        def test_active_duplicate_hash_reset_and_capacity(self):
            pool = native(4, True, "two_queue")
            ids = [b.block_id for b in pool.get_new_blocks(3)]
            cache(pool, ids, [8, 8, 9])
            pool.touch([pool.blocks[1], pool.null_block])
            pool.free_blocks([pool.blocks[1], pool.blocks[2], pool.blocks[3]])
            self.assertFalse(pool.reset_prefix_cache())
            before = state(pool)
            with self.assertRaises(ValueError):
                pool.get_new_blocks(3)
            self.assertEqual(before, state(pool))
            allocated = pool.get_new_blocks(2)
            self.assertNotIn(1, [b.block_id for b in allocated])
            self.assertEqual(
                pool.get_cached_block(utils.BlockHashType(8, (8,) * 16)).block_id, 1
            )
            pool.free_blocks([pool.blocks[1], *allocated])
            self.assertTrue(pool.reset_prefix_cache())
            self.assertEqual(pool.policy_stats().protected, 0)
            invariants(pool)

        def test_idle_transfer_quota_all_hot_and_metrics(self):
            pool = native(5, True, "two_queue", 1)
            blocks = pool.get_new_blocks(4)
            cache(pool, [b.block_id for b in blocks])
            pool.free_blocks(reversed(blocks))
            for b in blocks:
                pool.eviction_policy.promote(b)
            self.assertEqual(pool.policy_stats().hot_free, 4)
            before = state(pool)
            snapshot = pool.policy_stats()
            self.assertEqual(snapshot.promotions, 4)
            self.assertEqual(state(pool), before)
            pool.eviction_policy.get_all_free_blocks = lambda: (_ for _ in ()).throw(
                AssertionError("metrics must not traverse")
            )
            self.assertEqual(pool.policy_stats(), snapshot)
            allocated = pool.get_new_blocks(4)
            self.assertEqual(pool.policy_stats().hot_evictions, 4)
            pool.free_blocks(allocated)
            self.assertTrue(pool.reset_prefix_cache())
            self.assertEqual(pool.policy_stats().hot_evictions, 4)
            pool = native(5, True, "two_queue", 0.25)
            blocks = pool.get_new_blocks(4)
            cache(pool, [b.block_id for b in blocks])
            pool.touch(blocks)
            self.assertEqual(pool.policy_stats().demotions, 3)
            self.assertEqual(pool.policy_stats().protected, 1)

        def test_prometheus_record_consumes_lifetime_deltas_once(self):
            m = types.ModuleType("agentcache_logger_extracted")
            m.__dict__.update(StatLoggerBase=object)
            load_ast(m, ROOT / "vllm/v1/metrics/loggers.py", ("PrometheusStatLogger",))

            class Metric:
                value = 0

                def set(self, value):
                    self.value = value

                def inc(self, value):
                    self.value += value

            obj = object.__new__(m.PrometheusStatLogger)
            for name in (
                "gauge_scheduler_running",
                "gauge_scheduler_waiting",
                "gauge_gpu_cache_usage",
                "counter_gpu_prefix_cache_queries",
                "counter_gpu_prefix_cache_hits",
            ):
                setattr(obj, name, Metric())
            obj.policy_gauges = {
                f: Metric()
                for f in ("cold_free", "hot_free", "protected", "protected_capacity")
            }
            obj.policy_counters = {
                f: Metric()
                for f in ("promotions", "demotions", "cold_evictions", "hot_evictions")
            }
            obj._last_policy_counters = dict.fromkeys(obj.policy_counters, 0)
            pool = native(5, True, "two_queue")
            blocks = pool.get_new_blocks(4)
            cache(pool, [b.block_id for b in blocks])
            pool.touch(blocks)
            stats = types.SimpleNamespace(
                num_running_reqs=1,
                num_waiting_reqs=0,
                gpu_cache_usage=0.5,
                gpu_cache_policy=pool.policy_stats(),
                prefix_cache_stats=types.SimpleNamespace(queries=0, hits=0),
                spec_decoding_stats=None,
            )
            before = state(pool)
            obj.record(stats, None)
            obj.record(stats, None)
            self.assertEqual(obj.policy_counters["promotions"].value, 4)
            self.assertEqual(obj.policy_counters["demotions"].value, 2)
            self.assertEqual(state(pool), before)
            pool.free_blocks([b for b in blocks for _ in range(2)])
            self.assertTrue(pool.reset_prefix_cache())
            stats.gpu_cache_policy = pool.policy_stats()
            obj.record(stats, None)
            self.assertEqual(obj.policy_counters["promotions"].value, 4)

    return Contracts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", action="store_true")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as out:
        try:
            native, utils, policy, config = modules(args.extracted)
            stock = stock_class(args.archive, utils)
            frozen = frozen_class(args.reference, stock, utils)
            tests = build_tests(native, utils, policy, config, stock, frozen)
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(tests)
            )
            report = {
                "passed": result.wasSuccessful(),
                "scope": "extracted metadata"
                if args.extracted
                else "installed native CPU imports (not GPU)",
                "tests": result.testsRun,
                "compared_states": tests.compared_states,
                "failures": len(result.failures),
                "errors": len(result.errors),
            }
        except BaseException as error:
            json.dump({"passed": False, "error": repr(error)}, out)
            raise
        json.dump(report, out, indent=2)
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
