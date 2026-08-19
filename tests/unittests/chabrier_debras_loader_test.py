"""Download and cache tests for the Chabrier-Debras table loader."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Dict

import pytest

import exoeos.chabrier_debras as chabrier_debras
from exoeos import ChabrierDebrasEOS, ChabrierDebrasTableLoader


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_archive(path: Path, tables: Dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for filename, content in tables.items():
            member = tarfile.TarInfo(f"DirEOS2021/{filename}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _configure_local_archive(
    monkeypatch: pytest.MonkeyPatch,
    archive_path: Path,
    variant: str,
    tables: Dict[str, bytes],
) -> None:
    _write_archive(archive_path, tables)
    monkeypatch.setattr(chabrier_debras, "_ARCHIVE_URL", archive_path.as_uri())
    monkeypatch.setattr(
        chabrier_debras,
        "_ARCHIVE_SHA256",
        _digest(archive_path.read_bytes()),
    )
    monkeypatch.setitem(
        chabrier_debras._TABLE_CHECKSUMS,
        variant,
        tuple(_digest(tables[name]) for name in tables),
    )


def test_loader_exposes_published_metadata(tmp_path: Path) -> None:
    loader = ChabrierDebrasTableLoader(cache_directory=tmp_path)

    assert loader.variant == "Y0275"
    assert loader.cache_directory == tmp_path
    assert loader.expected_filenames == (
        "TABLEEOS_2021_TP_Y0275_v1",
        "TABLEEOS_2021_Trho_Y0275_v1",
    )
    assert loader.checksum == (
        "45f316790ce20d5d1ce0abee4db308521b5bfdc5526d0997141a2784834feeff"
    )
    assert loader.archive_checksum == loader.checksum
    assert loader.checksums == {
        "TABLEEOS_2021_TP_Y0275_v1": (
            "c4995d114affedddf421b57b847ad4872699e9526f32b4b335013ffcbfb0b938"
        ),
        "TABLEEOS_2021_Trho_Y0275_v1": (
            "b9ffd42d50c83cd691f3152fd21009b366af7044221cd291d87e5e5a48cc8299"
        ),
    }
    assert "Chabrier" in loader.citation
    assert "Debras" in loader.citation
    assert "10.3847/1538-4357/abfc48" in loader.citation
    assert loader.table_domain == {
        "temperature_K": (1.0e2, 1.0e8),
        "pressure_Pa": (1.0, 1.0e22),
        "mass_density_kg_m3": (1.0e-3, 1.0e9),
    }


def test_valid_cache_skips_download_and_load_delegates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loader = ChabrierDebrasTableLoader(cache_directory=tmp_path)
    contents = (b"cached TP table", b"cached T-rho table")
    for filename, content in zip(loader.expected_filenames, contents):
        (tmp_path / filename).write_bytes(content)
    monkeypatch.setitem(
        chabrier_debras._TABLE_CHECKSUMS,
        loader.variant,
        tuple(_digest(content) for content in contents),
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("A valid cache must not be downloaded again.")

    monkeypatch.setattr(chabrier_debras, "urlopen", fail_download)

    assert loader.fetch() == tmp_path

    sentinel = object()
    calls = []

    def fake_from_directory(cls, directory, *, variant):
        calls.append((Path(directory), variant))
        return sentinel

    monkeypatch.setattr(
        ChabrierDebrasEOS,
        "from_directory",
        classmethod(fake_from_directory),
    )

    assert loader.load() is sentinel
    assert calls == [(tmp_path, "Y0275")]


def test_missing_tables_are_downloaded_and_extracted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    loader = ChabrierDebrasTableLoader(
        variant="Y0292",
        cache_directory=cache_directory,
    )
    contents = (b"downloaded TP table", b"downloaded T-rho table")
    tables = dict(zip(loader.expected_filenames, contents))
    _configure_local_archive(
        monkeypatch,
        tmp_path / "tables.tar.gz",
        loader.variant,
        tables,
    )

    assert loader.fetch() == cache_directory
    assert (
        tuple(
            (cache_directory / filename).read_bytes()
            for filename in loader.expected_filenames
        )
        == contents
    )
    assert set(cache_directory.iterdir()) == {
        cache_directory / filename for filename in loader.expected_filenames
    }


def test_archive_checksum_failure_leaves_no_partial_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    loader = ChabrierDebrasTableLoader(cache_directory=cache_directory)
    tables = dict(zip(loader.expected_filenames, (b"TP table", b"T-rho table")))
    _configure_local_archive(
        monkeypatch,
        tmp_path / "tables.tar.gz",
        loader.variant,
        tables,
    )
    monkeypatch.setattr(chabrier_debras, "_ARCHIVE_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        loader.fetch()

    assert not any(
        (cache_directory / filename).exists() for filename in loader.expected_filenames
    )
    assert not list(cache_directory.glob(".DirEOS2021.*.tar.gz"))


def test_default_cache_directory_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    loader = ChabrierDebrasTableLoader()

    assert loader.cache_directory == tmp_path / "exoeos" / "DirEOS2021"


def test_invalid_variant_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        ChabrierDebrasTableLoader(
            variant="Y0280",
            cache_directory=tmp_path,
        )
