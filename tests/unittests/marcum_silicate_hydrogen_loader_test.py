"""Parsing, download, and cache tests for the Marcum table loader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import URLError

import jax.numpy as jnp
import numpy as np
import pytest

import exoeos.marcum_silicate_hydrogen as marcum
from exoeos import MarcumSilicateHydrogenEOS, MarcumSilicateHydrogenTableLoader


HEADER = (
    "H_wt_pct,P_GPa,T_K,rho_gcc,H_kJ_g,S_J_g_K,alpha_1e5,"
    "cp_J_g_K,KS_GPa,rho0_gcc,eta,gamma"
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _configure_tiny_published_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the native axes while preserving the published ragged layout."""

    monkeypatch.setattr(
        marcum,
        "_HYDROGEN_WEIGHT_PERCENT_AXIS",
        np.asarray([0.000100, 3.860990]),
    )
    monkeypatch.setattr(
        marcum,
        "_ENDMEMBER_FRACTION_AXIS",
        np.asarray([2.5e-5, 1.0]),
    )
    monkeypatch.setattr(marcum, "_TEMPERATURE_K_AXIS", np.asarray([6000.0, 6050.0]))
    monkeypatch.setattr(marcum, "_PRESSURE_GPA_AXIS", np.asarray([1.0, 5.0]))
    monkeypatch.setattr(marcum, "_COMPOSITION_COUNT", 2)
    monkeypatch.setattr(marcum, "_TEMPERATURE_COUNT", 2)
    monkeypatch.setattr(marcum, "_PRESSURE_COUNT", 2)
    monkeypatch.setattr(marcum, "_LOW_PRESSURE_COUNT", 1)
    monkeypatch.setattr(marcum, "_ROW_COUNT", 6)


def _tiny_rows() -> np.ndarray:
    rows = []
    weights = (0.000100, 3.860990)
    for composition_index, weight_percent in enumerate(weights):
        for temperature_index, temperature in enumerate((6000.0, 6050.0)):
            pressures = (1.0, 5.0) if temperature == 6000.0 else (5.0,)
            for pressure in pressures:
                field_base = (
                    1.0
                    + composition_index
                    + 0.1 * temperature_index
                    + 0.01 * pressure
                )
                fields = field_base + np.arange(9.0)
                rows.append((weight_percent, pressure, temperature, *fields))
    return np.asarray(rows)


def _write_csv(path: Path, rows: np.ndarray, header: str = HEADER) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        stream.write(header)
        stream.write("\n")
        np.savetxt(stream, rows, delimiter=",", fmt="%.12g")


def test_from_file_parses_ragged_grid_and_converts_to_si(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_published_grid(monkeypatch)
    table_path = tmp_path / "MgSiO3-H_lookup.csv"
    rows = _tiny_rows()
    _write_csv(table_path, rows)

    eos = MarcumSilicateHydrogenEOS.from_file(table_path)
    state = eos.state_tp(6050.0, 5.0e9, jnp.asarray([0.0, 1.0]))
    raw = rows[-1, 3:]
    expected = jnp.asarray(
        [
            5.0e9,
            raw[0] * 1.0e3,
            raw[1] * 1.0e6,
            raw[2] * 1.0e3,
            raw[3] * 1.0e-5,
            raw[4] * 1.0e3,
            raw[5] * 1.0e9,
            raw[6] * 1.0e3,
            raw[7],
            raw[8],
        ]
    )

    assert eos.fields.shape == (2, 2, 2, 9)
    assert jnp.all(jnp.isnan(eos.fields[:, 1, 0]))
    assert jnp.allclose(jnp.asarray(state), expected, rtol=1.0e-12)


def test_from_file_rejects_a_malformed_header(tmp_path: Path) -> None:
    table_path = tmp_path / "bad-header.csv"
    _write_csv(table_path, _tiny_rows(), header="wrong,header")

    with pytest.raises(ValueError, match="header"):
        MarcumSilicateHydrogenEOS.from_file(table_path)


def test_from_file_rejects_an_incomplete_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_published_grid(monkeypatch)
    table_path = tmp_path / "incomplete.csv"
    _write_csv(table_path, _tiny_rows()[:-1])

    with pytest.raises(ValueError, match="data rows"):
        MarcumSilicateHydrogenEOS.from_file(table_path)


def test_from_file_rejects_a_duplicate_grid_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_published_grid(monkeypatch)
    table_path = tmp_path / "duplicate.csv"
    rows = _tiny_rows()
    rows[1] = rows[0]
    _write_csv(table_path, rows)

    with pytest.raises(ValueError, match="pressure grid"):
        MarcumSilicateHydrogenEOS.from_file(table_path)


def test_from_file_rejects_a_misordered_composition_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_published_grid(monkeypatch)
    table_path = tmp_path / "misordered.csv"
    rows = _tiny_rows()
    rows[[0, 3]] = rows[[3, 0]]
    _write_csv(table_path, rows)

    with pytest.raises(ValueError, match="hydrogen grid"):
        MarcumSilicateHydrogenEOS.from_file(table_path)


def test_from_file_rejects_nonfinite_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_published_grid(monkeypatch)
    table_path = tmp_path / "nonfinite.csv"
    rows = _tiny_rows()
    rows[0, 3] = np.nan
    _write_csv(table_path, rows)

    with pytest.raises(ValueError, match="non-finite"):
        MarcumSilicateHydrogenEOS.from_file(table_path)


def test_loader_exposes_pinned_published_metadata(tmp_path: Path) -> None:
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=tmp_path)

    assert loader.cache_directory == tmp_path
    assert loader.expected_filename == "MgSiO3-H_lookup.csv"
    assert loader.checksum == (
        "153e6261008663bd2923cf54a85d7f3a11cc4303a2a6b8bbf38714a4d78fe69f"
    )
    assert loader.table_checksum == loader.checksum
    assert loader.commit == "5179f25721bd0bcc0827c7f77b5e24c18c36aa6f"
    assert loader.table_url.endswith(f"/{loader.commit}/{loader.expected_filename}")
    assert "Marcum" in loader.citation
    assert "Stixrude" in loader.citation
    assert "Young" in loader.citation
    assert "2608.27401" in loader.citation
    assert loader.table_domain == {
        "temperature_K": (3000.0, 10000.0),
        "pressure_Pa": (1.0e9, 8.0e11),
        "hydrogen_weight_percent": (0.000100, 3.860990),
        "hydrogen_endmember_mole_fraction": (2.5e-5, 1.0),
    }


def test_valid_cache_skips_download_and_load_delegates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=tmp_path)
    content = b"cached Marcum table"
    table_path = tmp_path / loader.expected_filename
    table_path.write_bytes(content)
    monkeypatch.setattr(marcum, "_TABLE_SHA256", _digest(content))

    def fail_download(*args, **kwargs):
        raise AssertionError("A valid cache must not be downloaded again.")

    monkeypatch.setattr(marcum, "urlopen", fail_download)

    assert loader.fetch() == table_path

    sentinel = object()
    calls = []

    def fake_from_file(cls, path):
        calls.append(Path(path))
        return sentinel

    monkeypatch.setattr(
        MarcumSilicateHydrogenEOS,
        "from_file",
        classmethod(fake_from_file),
    )

    assert loader.load() is sentinel
    assert calls == [table_path]


def test_missing_table_is_downloaded_and_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"downloaded Marcum table"
    source_path = tmp_path / "source.csv"
    source_path.write_bytes(content)
    cache_directory = tmp_path / "cache"
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=cache_directory)
    monkeypatch.setattr(marcum, "_TABLE_URL", source_path.as_uri())
    monkeypatch.setattr(marcum, "_TABLE_SHA256", _digest(content))

    table_path = loader.fetch()

    assert table_path == cache_directory / loader.expected_filename
    assert table_path.read_bytes() == content
    assert set(cache_directory.iterdir()) == {table_path}


def test_checksum_failure_leaves_no_partial_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.csv"
    source_path.write_bytes(b"corrupt table")
    cache_directory = tmp_path / "cache"
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=cache_directory)
    monkeypatch.setattr(marcum, "_TABLE_URL", source_path.as_uri())
    monkeypatch.setattr(marcum, "_TABLE_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        loader.fetch()

    assert not (cache_directory / loader.expected_filename).exists()
    assert not list(cache_directory.glob(f".{loader.expected_filename}.*"))


def test_download_failure_leaves_no_partial_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=cache_directory)

    def fail_download(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(marcum, "urlopen", fail_download)

    with pytest.raises(URLError, match="offline"):
        loader.fetch()

    assert not (cache_directory / loader.expected_filename).exists()
    assert not list(cache_directory.glob(f".{loader.expected_filename}.*"))


def test_default_cache_directory_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    loader = MarcumSilicateHydrogenTableLoader()

    assert loader.cache_directory == tmp_path / "exoeos" / "MgSiO3-H-EOS"
