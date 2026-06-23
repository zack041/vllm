# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-prefix-hit-first schedulers for local automatic prefix caching.

Run the async variant, which matches vLLM's default scheduling mode:

    PYTHONPATH=examples/features/automatic_prefix_caching \
    vllm serve MODEL \
        --enable-prefix-caching \
        --scheduler-cls longest_prefix_hit_scheduler.LongestPrefixHitScheduler

Use ``LongestPrefixHitSyncScheduler`` together with
``--no-async-scheduling`` for synchronous scheduling.
"""

import os
from typing import TypeVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

_SchedulerT = TypeVar("_SchedulerT", bound=Scheduler)
_DECODE_RESERVE_ENV = "VLLM_LONGEST_PREFIX_DECODE_RESERVE_TOKENS"


class _LongestPrefixHitMixin:
    def __init__(self: _SchedulerT, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError(
                "Longest-prefix-hit-first scheduling requires --scheduling-policy fcfs."
            )
        if self.connector is not None:
            raise ValueError(
                "Longest-prefix-hit-first scheduling only supports local KV cache."
            )
        if not self.kv_cache_manager.enable_caching:
            raise ValueError(
                "Longest-prefix-hit-first scheduling requires --enable-prefix-caching."
            )
        self._base_watermark_blocks = self.kv_cache_manager.watermark_blocks
        self._decode_reserve_tokens = int(os.getenv(_DECODE_RESERVE_ENV, "0"))
        if self._decode_reserve_tokens < 0:
            raise ValueError(f"{_DECODE_RESERVE_ENV} must be non-negative.")

    def _update_decode_reserve(self: _SchedulerT) -> None:
        """Reserve decode growth headroom when admitting a waiting request."""
        reserve_blocks_per_request = (
            self._decode_reserve_tokens + self.block_size - 1
        ) // self.block_size
        self.kv_cache_manager.watermark_blocks = (
            self._base_watermark_blocks
            + reserve_blocks_per_request * (len(self.running) + 1)
        )

    def _get_prefix_hit_tokens(self: _SchedulerT, request: Request) -> int:
        if (
            request.num_computed_tokens != 0
            or request.skip_reading_prefix_cache
            or not self.kv_cache_manager.enable_caching
        ):
            return 0

        _, num_computed_tokens = (
            self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes,
                request.num_tokens - 1,
            )
        )
        return num_computed_tokens

    def _select_waiting_queue_for_scheduling(
        self: _SchedulerT,
    ) -> RequestQueue | None:
        # Preserve the base scheduler's blocked-request promotion semantics.
        if self.skipped_waiting:
            self._update_decode_reserve()
            return self.skipped_waiting
        if not self.waiting:
            return None

        self._update_decode_reserve()
        request = min(
            self.waiting,
            key=lambda req: (
                -self._get_prefix_hit_tokens(req),
                req.arrival_time,
                req.request_id,
            ),
        )
        self.waiting.remove_request(request)
        self.waiting.prepend_request(request)
        return self.waiting


class LongestPrefixHitScheduler(_LongestPrefixHitMixin, AsyncScheduler):
    """Async scheduler that admits the waiting request with the longest hit."""


class LongestPrefixHitSyncScheduler(_LongestPrefixHitMixin, Scheduler):
    """Sync scheduler that admits the waiting request with the longest hit."""
