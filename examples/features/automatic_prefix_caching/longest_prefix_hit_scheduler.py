# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-prefix-hit-first schedulers for local automatic prefix caching.

Run the async variant, which matches vLLM's default scheduling mode:

    PYTHONPATH=examples/features/automatic_prefix_caching \
    vllm serve MODEL \
        --enable-prefix-caching \
        --scheduler-cls longest_prefix_hit_scheduler.LongestPrefixHitScheduler

Set ``VLLM_LONGEST_PREFIX_BUFFER_PERCENT`` to tune the stale-prefix-cache
buffer for waiting-request admission. It defaults to 5% of vLLM's usable KV
block pool. Existing running requests can still decode into this buffer.

This experiment intentionally does not bypass the buffer for oversized idle
admissions. If a request's known sequence cannot fit within the non-buffered
KV pool, lower the buffer percentage or disable it.

Use ``LongestPrefixHitSyncScheduler`` together with
``--no-async-scheduling`` for synchronous scheduling.
"""

import math
import os
from typing import TypeVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

_SchedulerT = TypeVar("_SchedulerT", bound=Scheduler)
_PREFIX_BUFFER_PERCENT_ENV = "VLLM_LONGEST_PREFIX_BUFFER_PERCENT"


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
        self._prefix_buffer_percent = float(os.getenv(_PREFIX_BUFFER_PERCENT_ENV, "5"))
        if not 0 <= self._prefix_buffer_percent <= 100:
            raise ValueError(f"{_PREFIX_BUFFER_PERCENT_ENV} must be between 0 and 100.")
        self._install_waiting_buffer_admission_gate()

    def _install_waiting_buffer_admission_gate(self: _SchedulerT) -> None:
        allocate_slots = self.kv_cache_manager.allocate_slots

        def allocate_slots_with_waiting_buffer(request: Request, *args, **kwargs):
            if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                kwargs["has_scheduled_reqs"] = True
            return allocate_slots(request, *args, **kwargs)

        self.kv_cache_manager.allocate_slots = allocate_slots_with_waiting_buffer

    def _update_waiting_admission_watermark(self: _SchedulerT) -> None:
        """Protect stale prefix-cache blocks when admitting waiting requests."""
        usable_pool_blocks = max(self.kv_cache_manager.block_pool.num_gpu_blocks - 1, 0)
        prefix_buffer_blocks = math.ceil(
            usable_pool_blocks * self._prefix_buffer_percent / 100
        )
        self.kv_cache_manager.watermark_blocks = (
            self._base_watermark_blocks + prefix_buffer_blocks
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
            self._update_waiting_admission_watermark()
            return self.skipped_waiting
        if not self.waiting:
            return None

        self._update_waiting_admission_watermark()
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
