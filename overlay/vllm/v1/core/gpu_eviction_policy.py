# SPDX-License-Identifier: Apache-2.0
"""Bounded Two-Queue idle-block policy, not ARC (no ghost/adaptive history)."""

import math
from collections import OrderedDict
from typing import Protocol

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.metrics.stats import GPUCachePolicyStats


def validate_policy(name: str, ratio: float, enable_caching: bool) -> None:
    if name not in ("lru", "two_queue"):
        raise ValueError(f"Unknown GPU eviction policy: {name}")
    if name == "two_queue":
        if not enable_caching:
            raise ValueError("two_queue requires automatic prefix caching")
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("GPU protected ratio must be finite and in (0, 1]")


class GPUCachePolicy(Protocol):
    """Policy hooks; block ownership, references and hash indexes stay native."""

    def promote(self, block: KVCacheBlock) -> None: ...
    def forget(self, block: KVCacheBlock) -> None: ...
    def reset_policy(self) -> None: ...
    def snapshot(self) -> GPUCachePolicyStats: ...


class TwoQueuePolicy:
    """Free queue and lifecycle hooks equivalent to frozen experimental v2.

    The protected quota includes active blocks. Demotion changes priority,
    never reference counts. Allocation candidates are exclusively idle blocks.
    """

    def __init__(self, blocks: list[KVCacheBlock], capacity: int):
        def empty_queue():
            queue = FreeKVCacheBlockQueue([KVCacheBlock(-1)])
            queue.popleft()
            return queue

        self.cold = empty_queue()
        self.hot = empty_queue()
        self.members: dict[int, FreeKVCacheBlockQueue] = {}
        self.protected: OrderedDict[int, KVCacheBlock] = OrderedDict()
        self.protected_capacity = capacity
        self.promotions = 0
        self.demotions = 0
        self.cold_evictions = 0
        self.hot_evictions = 0
        for block in blocks:
            self.append(block)

    @property
    def num_free_blocks(self) -> int:
        return self.cold.num_free_blocks + self.hot.num_free_blocks

    def popleft(self) -> KVCacheBlock:
        queue = self.cold if self.cold.num_free_blocks else self.hot
        block = queue.popleft()
        del self.members[block.block_id]
        return block

    def remove(self, block: KVCacheBlock) -> None:
        self.members.pop(block.block_id).remove(block)

    def append(self, block: KVCacheBlock) -> None:
        if block.block_id in self.members or block.ref_cnt != 0:
            raise RuntimeError("Only unqueued idle blocks may enter free queue")
        if block.block_hash is None:
            self.forget(block)
        queue = self.hot if block.block_id in self.protected else self.cold
        queue.append(block)
        self.members[block.block_id] = queue

    def promote(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            return
        block_id = block.block_id
        if block_id not in self.protected:
            self.promotions += 1
        self.protected[block_id] = block
        self.protected.move_to_end(block_id)
        if self.members.get(block_id) is self.cold:
            if block.ref_cnt != 0:
                raise RuntimeError("Active blocks cannot change free segment")
            self.cold.remove(block)
            self.hot.append(block)
            self.members[block_id] = self.hot
        if len(self.protected) > self.protected_capacity:
            demoted_id, demoted = self.protected.popitem(last=False)
            self.demotions += 1
            if self.members.get(demoted_id) is self.hot:
                if demoted.ref_cnt != 0:
                    raise RuntimeError("Active blocks cannot change free segment")
                self.hot.remove(demoted)
                self.cold.append(demoted)
                self.members[demoted_id] = self.cold

    def forget(self, block: KVCacheBlock) -> None:
        self.protected.pop(block.block_id, None)

    def record_eviction(self, was_hot: bool) -> None:
        if was_hot:
            self.hot_evictions += 1
        else:
            self.cold_evictions += 1

    def reset_policy(self) -> None:
        self.protected.clear()
        while self.hot.num_free_blocks:
            block = self.hot.popleft()
            del self.members[block.block_id]
            self.append(block)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Debug/test traversal only; never called by metrics export."""
        return self.cold.get_all_free_blocks() + self.hot.get_all_free_blocks()

    def snapshot(self) -> GPUCachePolicyStats:
        return GPUCachePolicyStats(
            cold_free=self.cold.num_free_blocks,
            hot_free=self.hot.num_free_blocks,
            protected=len(self.protected),
            protected_capacity=self.protected_capacity,
            promotions=self.promotions,
            demotions=self.demotions,
            cold_evictions=self.cold_evictions,
            hot_evictions=self.hot_evictions,
        )
