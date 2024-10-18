"""Test the package itself."""

import importlib.metadata

import spexial


def test_version():
    """Test version."""
    assert importlib.metadata.version("spexial") == spexial.__version__
