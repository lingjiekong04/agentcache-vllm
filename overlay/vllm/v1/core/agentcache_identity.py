# SPDX-License-Identifier: Apache-2.0
"""Startup evidence from the actual EngineCore, not the launching shell."""

import hashlib
import json
import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def log_engine_identity(config, pool) -> None:
    root = Path(__file__).resolve().parents[2]
    files = (
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
    )
    cache = config.cache_config
    from vllm.platforms import current_platform

    gpu_uuid = None
    gpu_identity_error = None
    if current_platform.is_cuda():
        from vllm.platforms.cuda import device_id_to_physical_device_id
        from vllm.utils import import_pynvml

        try:
            nvml = import_pynvml()
            nvml.nvmlInit()
            try:
                physical = device_id_to_physical_device_id(0)
                handle = nvml.nvmlDeviceGetHandleByIndex(physical)
                gpu_uuid = nvml.nvmlDeviceGetUUID(handle)
                gpu_uuid = (
                    gpu_uuid.decode()
                    if isinstance(gpu_uuid, bytes)
                    else gpu_uuid
                )
            finally:
                nvml.nvmlShutdown()
        except Exception as error:
            # Observation must not break stock LRU serving. Acceptance tools
            # reject missing GPU identity rather than claiming a verified run.
            gpu_identity_error = repr(error)
            logger.warning("AgentCache GPU identity unavailable: %s", error)
    evidence = {
        "pid": os.getpid(),
        "run_id": os.environ.get("AGENTCACHE_RUN_ID", config.instance_id),
        "source_root": str(root),
        "gpu_uuid": gpu_uuid,
        "gpu_identity_error": gpu_identity_error,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy": cache.gpu_eviction_policy,
        "pool_type": f"{type(pool).__module__}.{type(pool).__qualname__}",
        "queue_type": (
            f"{type(pool.free_block_queue).__module__}."
            f"{type(pool.free_block_queue).__qualname__}"
        ),
        "config": {
            "model": config.model_config.model,
            "dtype": str(config.model_config.dtype),
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "blocks": pool.num_gpu_blocks,
            "block_size": cache.block_size,
            "protected_ratio": cache.gpu_cache_protected_ratio,
            "apc": cache.enable_prefix_caching,
            "seed": config.model_config.seed,
            "gpu_memory_utilization": cache.gpu_memory_utilization,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
            "data_parallel_size": config.parallel_config.data_parallel_size,
        },
        "policy_stats": vars(pool.policy_stats()),
        "source_sha256": {
            file: hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in files
        },
    }
    logger.info(
        "AGENTCACHE_NATIVE_ENGINE %s", json.dumps(evidence, sort_keys=True)
    )
