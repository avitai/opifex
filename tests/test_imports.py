"""Framework-level import tests.

This module contains tests for basic package imports and framework setup.
"""


def test_import_opifex():
    """Test that the main opifex package can be imported."""
    import opifex

    assert opifex.__version__ == "0.1.0"


def test_import_neural_networks():
    """Test that neural network modules can be imported."""
    import opifex.neural

    # Test imports are successful
    assert hasattr(opifex.neural, "base")
