"""Create and manage connection pools."""

from __future__ import annotations

import atexit
from threading import Lock
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

from urllib3 import PoolManager, Retry
from pynldas2.exceptions import DownloadError

__all__ = ["HTTPSPool"]
CHUNK_SIZE = 1024 * 1024  # 1 MB

class HTTPSPool:
    """Singleton to manage an HTTP(S) connection pool using PoolManager."""

    _instance: ClassVar[PoolManager | None] = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> PoolManager:
        """Retrieve or create a shared PoolManager instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = PoolManager(
                        num_pools=10,
                        maxsize=10,
                        block=True,
                        retries=Retry(
                            total=5,
                            backoff_factor=0.5,
                            status_forcelist=[500, 502, 504],
                            allowed_methods=["HEAD", "GET", "POST"],
                        ),
                        timeout=10.0,
                    )
        return cls._instance

    @classmethod
    def download_files(cls, url_list: list[str], file_list: list[Path], rewrite: bool = False) -> None:
        """Download a list of files from corresponding URLs.

        Parameters
        ----------
        url_list : List[str]
            List of URLs to download files from.
        file_list : List[Path]
            List of file paths to save the downloaded content.
        rewrite : bool, optional
            If True, overwrite existing files; otherwise, skip downloading if the file exists and matches size.
        """
        if len(url_list) != len(file_list):
            raise ValueError("The number of URLs must match the number of file paths.")

        pool = cls.get_instance()
        max_workers = min(4, os.cpu_count() or 1, len(url_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(cls._download_single, pool, url, file, rewrite): url
                for url, file in zip(url_list, file_list)
            }
            for future in as_completed(future_to_url):
                try:
                    future.result()
                except Exception as e:
                    raise DownloadError(f"Error downloading {future_to_url[future]}: {e}") from e

    @staticmethod
    def _download_single(pool: PoolManager, url: str, output_file: Path, rewrite: bool) -> None:
        """
        Download a single file from a URL to the specified path.

        Parameters
        ----------
        pool : PoolManager
            The shared PoolManager instance.
        url : str
            The URL to download the file from.
        output_file : Path
            The file path to save the downloaded content.
        rewrite : bool
            If True, overwrite existing files.
        """
        head = pool.request("HEAD", url)
        fsize = int(head.headers.get("Content-Length", -1))
        if output_file.exists() and output_file.stat().st_size == fsize and not rewrite:
            return

        output_file.unlink(missing_ok=True)
        with open(output_file, "wb") as f:
            response = pool.request("GET", url, preload_content=False)
            for chunk in response.stream(CHUNK_SIZE):  # 1 MB chunks
                f.write(chunk)
            response.release_conn()

    @classmethod
    def close(cls):
        """Cleanup the PoolManager."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None


atexit.register(HTTPSPool.close)
