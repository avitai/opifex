"""Framework-level import tests.

This module contains tests for basic package imports and framework setup.
"""


def test_import_opifex():
    """Test that the main opifex package can be imported and reports its version.

    The version is read from installed package metadata in `opifex/__init__.py`,
    which falls back to "0.0.0+unknown" when that metadata cannot be found. This
    asserts the fallback did not fire rather than pinning a literal, so a release
    does not break the test.
    """
    import opifex

    assert opifex.__version__, "opifex.__version__ is empty"
    assert opifex.__version__ != "0.0.0+unknown", (
        "package metadata could not be read, so the version fell back to its sentinel"
    )


def test_import_neural_networks():
    """Test that neural network modules can be imported."""
    import opifex.neural

    # Test imports are successful
    assert hasattr(opifex.neural, "base")
