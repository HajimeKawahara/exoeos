"""Private helpers for downloading verified EOS tables."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_directory(name: str) -> Path:
    """Return the package cache directory for a named table set."""

    cache_root = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return root / "exoeos" / name


@contextmanager
def verified_download(
    *,
    url: str,
    checksum: str,
    cache_directory: Path,
    prefix: str,
    suffix: str = "",
    checksum_error: str,
    opener: Callable[..., Any],
) -> Iterator[Path]:
    """Download to a temporary file and yield it after checksum validation."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_directory,
            prefix=prefix,
            suffix=suffix,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            with opener(url, timeout=60) as source:
                shutil.copyfileobj(source, destination)
        if sha256(temporary_path) != checksum:
            raise ValueError(checksum_error)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_verified_file(
    *,
    url: str,
    checksum: str,
    cache_directory: Path,
    filename: str,
    checksum_error: str,
    opener: Callable[..., Any],
) -> Path:
    """Return a verified cached file, downloading and replacing atomically."""

    path = cache_directory / filename
    if path.is_file() and sha256(path) == checksum:
        return path

    with verified_download(
        url=url,
        checksum=checksum,
        cache_directory=cache_directory,
        prefix=f".{filename}.",
        checksum_error=checksum_error,
        opener=opener,
    ) as temporary_path:
        temporary_path.replace(path)
    return path
