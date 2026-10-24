# SPDX-License-Identifier: Apache-2.0
"""Equivalent v1 segmented policy, with cheaper idle segment transfers.

CPU experiment only: this module does not activate a serving replacement.
Equivalence concerns valid states maintained by BlockPool, not corrupted input.
Native linked-list operations, hash eviction, references and quota are unchanged.
"""

import hashlib
import os
from pathlib import Path

import policy as v1


V1_SHA256 = "c17f572d74c83695b8ff41700672cec5504d04f9700370f60832058a5e792b7f"


def validate_reference():
    actual = hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest()
    if actual != V1_SHA256:
        raise RuntimeError("v1 policy.py changed; refusing an unverified reference")


class SegmentedFreeQueueV2(v1.SegmentedFreeQueue):
    def promote(self, block):
        if block.block_hash is None:
            return
        block_id = block.block_id
        self.protected[block_id] = block
        self.protected.move_to_end(block_id)
        if self.members.get(block_id) is self.cold:
            if block.ref_cnt != 0:
                raise RuntimeError("Only idle blocks may move between free segments")
            # Same physical move as v1, without popping/reinserting membership
            # or rediscovering the destination through the generic append path.
            self.cold.remove(block)
            self.hot.append(block)
            self.members[block_id] = self.hot
        if len(self.protected) > self.protected_capacity:
            demoted_id, demoted = self.protected.popitem(last=False)
            if self.members.get(demoted_id) is self.hot:
                if demoted.ref_cnt != 0:
                    raise RuntimeError(
                        "Only idle blocks may move between free segments"
                    )
                self.hot.remove(demoted)
                self.cold.append(demoted)
                self.members[demoted_id] = self.cold


def make_pool_class(base_pool, queue_type, block_type, ratio, announce=False):
    validate_reference()
    reference_pool = v1.make_pool_class(base_pool, queue_type, block_type, ratio)

    class AgentCacheBlockPoolV2(reference_pool):
        agentcache_policy = "two_queue_v2"

        def __init__(self, num_gpu_blocks, enable_caching):
            # Construct the stock pool once, then install the v2 queue. Inherit
            # v1's touch, hash-eviction and reset methods without duplicating them.
            base_pool.__init__(self, num_gpu_blocks, enable_caching)
            if not enable_caching:
                return
            old_queue = self.free_block_queue
            blocks = []
            while old_queue.num_free_blocks:
                blocks.append(old_queue.popleft())
            capacity = max(1, int((num_gpu_blocks - 1) * ratio))
            self.free_block_queue = SegmentedFreeQueueV2(
                queue_type, block_type, blocks, capacity
            )
            if announce:
                print(
                    f"AGENTCACHE_BLOCK_POOL_ACTIVE pid={os.getpid()} "
                    f"policy=two_queue_v2 blocks={num_gpu_blocks} "
                    f"protected_capacity={capacity}",
                    flush=True,
                )

    return AgentCacheBlockPoolV2
