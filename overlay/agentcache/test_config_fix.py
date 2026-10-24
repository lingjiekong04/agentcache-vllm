"""Standard-library fixture/config checks; not native scheduler execution."""

import ast
import types
import unittest
from pathlib import Path

import check_cpu

ROOT = Path(__file__).resolve().parents[1]


def fixture_config(policy):
    module = types.ModuleType("config_fixture_test")
    module.__dict__.update(
        torch=types.SimpleNamespace(
            device=lambda name: ("device", name), float32="float32"
        ),
        CacheConfig=types.SimpleNamespace,
        ModelConfig=types.SimpleNamespace,
        SchedulerConfig=types.SimpleNamespace,
        VllmConfig=types.SimpleNamespace,
        KVCacheConfig=types.SimpleNamespace,
        KVCacheGroupSpec=lambda *args: args,
        FullAttentionSpec=lambda *args: args,
        StructuredOutputManager=lambda config: config,
        Scheduler=lambda *args, **kwargs: kwargs["structured_output_manager"],
    )
    check_cpu.load_ast(module, ROOT / "vllm/config.py", ("DeviceConfig",))
    check_cpu.load_ast(
        module, ROOT / "tests/v1/core/test_scheduler.py", ("create_scheduler",)
    )
    return module.create_scheduler(
        gpu_eviction_policy=policy, enable_prefix_caching=True
    )


class ConfigFix(unittest.TestCase):
    def test_lru_fixture_keeps_original_unspecified_device(self):
        config = fixture_config("lru")
        self.assertIsNone(config.device_config)
        self.assertEqual(config.cache_config.gpu_eviction_policy, "lru")

    def test_two_queue_fixture_supplies_real_cuda_device_config(self):
        config = fixture_config("two_queue")
        self.assertEqual(config.device_config.device_type, "cuda")
        self.assertEqual(config.device_config.device, ("device", "cuda"))
        self.assertEqual(config.cache_config.gpu_eviction_policy, "two_queue")

    def test_missing_configs_are_rejected_before_model_or_pool_creation(self):
        *_, cache_config = check_cpu.modules(True)
        cache = cache_config(
            16,
            0.2,
            0,
            "auto",
            enable_prefix_caching=True,
            gpu_eviction_policy="two_queue",
        )
        for missing, message in (
            ("device_config", "DeviceConfig"),
            ("parallel_config", "single GPU"),
        ):
            with self.subTest(missing=missing):
                config = types.SimpleNamespace(
                    device_config=types.SimpleNamespace(device_type="cuda"),
                    parallel_config=types.SimpleNamespace(
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        data_parallel_size=1,
                    ),
                )
                setattr(config, missing, None)
                with self.assertRaisesRegex(ValueError, message):
                    cache.verify_gpu_policy_with_vllm_config(config)

    def test_both_scheduler_contracts_remain_selected(self):
        tree = ast.parse(
            (ROOT / "tests/v1/core/test_scheduler.py").read_bytes()
        )
        names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn(
            "test_agentcache_cancel_shared_prefix_keeps_active_owner", names
        )
        self.assertIn(
            "test_agentcache_preemption_then_resume_releases_all_blocks", names
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
