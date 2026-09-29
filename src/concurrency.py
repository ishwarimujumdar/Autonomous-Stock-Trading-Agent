import asyncio
from collections.abc import Awaitable, Iterable

from src.config import MAX_CONCURRENCY


async def gather_limited(tasks: Iterable[Awaitable], limit: int = MAX_CONCURRENCY) -> list:
    """Runs the tasks together, but never more than `limit` at once. Results keep their order."""
    semaphore = asyncio.Semaphore(limit)

    async def run(task):
        async with semaphore:
            return await task

    return list(await asyncio.gather(*(run(t) for t in tasks)))
