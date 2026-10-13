"""Shared pytest configuration for deterministic JAX tests."""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Keep each file together unless a fixture defines a smaller JAX cache group."""
    for item in items:
        if item.get_closest_marker("xdist_group") is None:
            item.add_marker(pytest.mark.xdist_group(item.nodeid.split("::", 1)[0]))
