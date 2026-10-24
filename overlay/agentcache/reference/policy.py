# SPDX-License-Identifier: Apache-2.0
"""Opt-in segmented FIFO/LRU experiment for the supplied vLLM 0.8.3 snapshot.

This is not full ARC/2Q: no ghost history or adaptive partition is implemented.
The hot quota includes active and idle blocks. Only idle blocks are evictable.
"""

from collections import OrderedDict
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys


EXPECTED = {
    "block_pool.py": "82076397ed6eb65127cc8cca9ee8fd669fe39ca8e7d6ee87381e2771782a7748",
    "kv_cache_utils.py": "82857a94f1b54748287acabd3c3dd8791cbb374a300cb961bab8cc0213c61e63",
    "kv_cache_manager.py": "577529cb7ed73e0b882e969f1a355fcca734d621f6b085532d65b6e4ef5e9e46",
}


def validate_ratio(ratio):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("protected ratio must be finite and in (0, 1]")


class SegmentedFreeQueue:
    """Evict probationary blocks first, falling back to bounded hot blocks."""

    def __init__(self, queue_type, block_type, blocks, protected_capacity):
        def empty_queue():
            # The legacy constructor requires a nonempty list.
            queue = queue_type([block_type(-1)])
            queue.popleft()
            return queue

        self.cold = empty_queue()
        self.hot = empty_queue()
        self.members = {}
        self.protected = OrderedDict()
        self.protected_capacity = protected_capacity
        for block in blocks:
            self.append(block)

    @property
    def num_free_blocks(self):
        return self.cold.num_free_blocks + self.hot.num_free_blocks

    def popleft(self):
        queue = self.cold if self.cold.num_free_blocks else self.hot
        block = queue.popleft()
        del self.members[block.block_id]
        return block

    def remove(self, block):
        self.members.pop(block.block_id).remove(block)

    def append(self, block):
        if block.block_id in self.members or block.ref_cnt != 0:
            raise RuntimeError("Only unqueued, idle blocks may enter the free queue")
        if block.block_hash is None:
            self.forget(block)
        queue = self.hot if block.block_id in self.protected else self.cold
        queue.append(block)
        self.members[block.block_id] = queue

    def promote(self, block):
        if block.block_hash is None:
            return
        self.protected[block.block_id] = block
        self.protected.move_to_end(block.block_id)
        if self.members.get(block.block_id) is self.cold:
            self.remove(block)
            self.append(block)
        if len(self.protected) > self.protected_capacity:
            _, demoted = self.protected.popitem(last=False)
            if self.members.get(demoted.block_id) is self.hot:
                self.remove(demoted)
                self.append(demoted)

    def forget(self, block):
        self.protected.pop(block.block_id, None)

    def reset_policy(self):
        self.protected.clear()
        while self.hot.num_free_blocks:
            block = self.hot.popleft()
            del self.members[block.block_id]
            self.append(block)

    def get_all_free_blocks(self):
        return self.cold.get_all_free_blocks() + self.hot.get_all_free_blocks()


def make_pool_class(base_pool, queue_type, block_type, ratio, announce=False):
    """Keep base methods, overriding only queue choice and policy lifecycle."""
    validate_ratio(ratio)

    class AgentCacheBlockPool(base_pool):
        agentcache_policy = "two_queue"

        def __init__(self, num_gpu_blocks, enable_caching):
            super().__init__(num_gpu_blocks, enable_caching)
            if not enable_caching:
                return
            old_queue = self.free_block_queue
            blocks = []
            while old_queue.num_free_blocks:
                blocks.append(old_queue.popleft())
            capacity = max(1, int((num_gpu_blocks - 1) * ratio))
            self.free_block_queue = SegmentedFreeQueue(
                queue_type, block_type, blocks, capacity
            )
            if announce:
                print(
                    f"AGENTCACHE_BLOCK_POOL_ACTIVE pid={os.getpid()} "
                    f"policy=two_queue blocks={num_gpu_blocks} "
                    f"protected_capacity={capacity}",
                    flush=True,
                )

        def touch(self, blocks):
            super().touch(blocks)
            if self.enable_caching:
                for block in blocks:
                    if block is not self.null_block:
                        self.free_block_queue.promote(block)

        def _maybe_evict_cached_block(self, block):
            self.free_block_queue.forget(block)
            return super()._maybe_evict_cached_block(block)

        def reset_prefix_cache(self):
            success = super().reset_prefix_cache()
            if success and self.enable_caching:
                self.free_block_queue.reset_policy()
            return success

    return AgentCacheBlockPool


def load_installed():
    """Fail closed if the installed version/source differs from this experiment."""
    if version("vllm") != "0.8.3":
        raise RuntimeError("This experiment supports vLLM 0.8.3 only")
    from vllm.v1.core import block_pool, kv_cache_utils, kv_cache_manager

    root = Path(block_pool.__file__).parent
    for filename, expected in EXPECTED.items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source mismatch: {filename}; refusing to patch")
    return block_pool, kv_cache_utils, kv_cache_manager


def activate(ratio):
    """Apply a process-local replacement; never write site-packages files."""
    pool_module, utils_module, manager_module = load_installed()
    existing = pool_module.BlockPool
    if getattr(existing, "agentcache_policy", None) == "two_queue":
        return
    replacement = make_pool_class(
        existing,
        utils_module.FreeKVCacheBlockQueue,
        utils_module.KVCacheBlock,
        ratio,
        announce=True,
    )
    # Imports may already have bound the original class in module globals.
    for name in (
        "vllm.v1.core.block_pool",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.specialized_manager",
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "BlockPool", None) is existing:
            module.BlockPool = replacement
    assert manager_module.BlockPool is replacement
