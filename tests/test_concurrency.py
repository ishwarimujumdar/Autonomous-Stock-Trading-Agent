import asyncio

import pytest

from src.concurrency import gather_limited


@pytest.mark.asyncio
async def test_never_runs_more_than_the_limit_at_once():
    running, peak = 0, 0

    async def task(n):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return n

    await gather_limited((task(n) for n in range(10)), limit=2)
    assert peak == 2


@pytest.mark.asyncio
async def test_results_keep_their_original_order():
    async def task(n):
        await asyncio.sleep(0.03 - n * 0.01)  # later tasks finish first
        return n

    assert await gather_limited((task(n) for n in range(3)), limit=3) == [0, 1, 2]
