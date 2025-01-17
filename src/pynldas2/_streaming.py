from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Sequence

from aiohttp import TCPConnector
from aiohttp_client_cache import CachedSession
from aiohttp.client_exceptions import ClientResponseError

if sys.platform == "win32":  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

__all__ = ["fetch_binary"]
CHUNK_SIZE = 1024 * 1024  # Default chunk size of 1 MB
MAX_HOSTS = 5  # Maximum connections to a single host (rate-limited service)
TIMEOUT = 30  # Timeout for requests in seconds

class FetchError(Exception):
    """Exception raised for fetch errors."""

    def __init__(self, err: str, url: str | None = None) -> None:
        self.message = (
            f"Service returned the following error:\nURL: {url}\nERROR: {err}" if url else err
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

async def _fetch_content(session: CachedSession, url: str) -> bytes:
    """Fetch the binary content of a URL."""
    try:
        async with session.get(url) as response:
            return await response.read()
    except (ClientResponseError, ValueError) as ex:
        raise FetchError(await response.text(), str(response.url)) from ex

async def _fetch_session(urls: Sequence[str]) -> list[bytes]:
    """Fetch binary content concurrently."""
    async with CachedSession(connector=TCPConnector(limit_per_host=MAX_HOSTS), raise_for_status=True, timeout=TIMEOUT) as session:
        tasks = [_fetch_content(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

def _get_or_create_event_loop() -> tuple[asyncio.AbstractEventLoop, bool]:
    """Retrieve or create an event loop."""
    with contextlib.suppress(RuntimeError):
        return asyncio.get_running_loop(), False
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    return new_loop, True

def fetch_binary(urls: Sequence[str]) -> list[bytes]:
    """Fetch binary content of multiple URLs concurrently.

    Parameters
    ----------
    urls : Sequence[str]
        Immutable list of URLs to fetch.

    Returns
    -------
    list[bytes]
        List of binary content for each URL.
    """
    loop, is_new_loop = _get_or_create_event_loop()

    try:
        return loop.run_until_complete(_fetch_session(urls))
    finally:
        if is_new_loop:
            loop.close()
