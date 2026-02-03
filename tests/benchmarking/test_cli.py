"""Tests for benchmarking CLI module.

TDD: Tests written FIRST before implementation.
"""

import pytest


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_parse_benchmark_flag(self):
        """CLI parses -b/--benchmark flag."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["-b", "PDEBench_2D_DarcyFlow"])
        assert args.benchmark == "PDEBench_2D_DarcyFlow"

        args = parse_args(["--benchmark", "PDEBench_1D_Burgers"])
        assert args.benchmark == "PDEBench_1D_Burgers"

    def test_parse_operator_flag(self):
        """CLI parses -o/--operator flag."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["-o", "TensorizedFourierNeuralOperator"])
        assert args.operator == "TensorizedFourierNeuralOperator"

    def test_parse_epochs_flag(self):
        """CLI parses --epochs flag with default."""
        from opifex.benchmarking.cli import parse_args

        # Default value
        args = parse_args(["-b", "test"])
        assert args.epochs == 100

        # Custom value
        args = parse_args(["-b", "test", "--epochs", "50"])
        assert args.epochs == 50

    def test_parse_batch_size_flag(self):
        """CLI parses --batch-size flag with default."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["-b", "test"])
        assert args.batch_size == 32

        args = parse_args(["-b", "test", "--batch-size", "16"])
        assert args.batch_size == 16

    def test_parse_output_flag(self):
        """CLI parses --output flag for results file."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["-b", "test", "--output", "results.json"])
        assert args.output == "results.json"

    def test_parse_list_benchmarks_flag(self):
        """CLI parses --list-benchmarks flag."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["--list-benchmarks"])
        assert args.list_benchmarks is True

    def test_parse_list_operators_flag(self):
        """CLI parses --list-operators flag."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["--list-operators"])
        assert args.list_operators is True

    def test_parse_seed_flag(self):
        """CLI parses --seed flag for reproducibility."""
        from opifex.benchmarking.cli import parse_args

        args = parse_args(["-b", "test", "--seed", "123"])
        assert args.seed == 123


class TestCLIListCommands:
    """Tests for CLI list commands."""

    def test_list_benchmarks_shows_pdebench(self, capsys):
        """--list-benchmarks shows registered benchmarks."""
        from opifex.benchmarking.cli import run_cli

        # Should exit with 0 after listing
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["--list-benchmarks"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "PDEBench_2D_DarcyFlow" in captured.out

    def test_list_operators_shows_tfno(self, capsys):
        """--list-operators shows registered operators."""
        from opifex.benchmarking.cli import run_cli

        with pytest.raises(SystemExit) as exc_info:
            run_cli(["--list-operators"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "TensorizedFourierNeuralOperator" in captured.out


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_invalid_benchmark_raises_error(self, capsys):
        """Invalid benchmark name returns non-zero exit code."""
        from opifex.benchmarking.cli import run_cli

        exit_code = run_cli(
            ["-b", "NonExistentBenchmark", "-o", "TensorizedFourierNeuralOperator"]
        )
        assert exit_code != 0

        captured = capsys.readouterr()
        assert (
            "NonExistentBenchmark" in captured.err
            or "not found" in captured.err.lower()
        )

    def test_invalid_operator_raises_error(self, capsys):
        """Invalid operator name returns non-zero exit code."""
        from opifex.benchmarking.cli import run_cli

        exit_code = run_cli(
            ["-b", "PDEBench_2D_DarcyFlow", "-o", "NonExistentOperator"]
        )
        assert exit_code != 0

        captured = capsys.readouterr()
        assert (
            "NonExistentOperator" in captured.err or "not found" in captured.err.lower()
        )

    def test_missing_required_args_shows_help(self, capsys):
        """Missing required args shows usage help."""
        from opifex.benchmarking.cli import run_cli

        # No benchmark or list flag - should show error
        with pytest.raises(SystemExit) as exc_info:
            run_cli([])
        # argparse exits with 2 for argument errors
        assert exc_info.value.code == 2


class TestCLIExecution:
    """Tests for CLI benchmark execution."""

    def test_run_benchmark_returns_result(self, capsys):
        """Running benchmark produces result output."""
        from opifex.benchmarking.cli import run_cli

        # Quick benchmark (2 epochs) for test speed
        exit_code = run_cli(
            [
                "-b",
                "PDEBench_2D_DarcyFlow",
                "-o",
                "TensorizedFourierNeuralOperator",
                "--epochs",
                "2",
                "--batch-size",
                "4",
            ]
        )

        assert exit_code == 0

        captured = capsys.readouterr()
        # Should show metrics
        assert "mse" in captured.out.lower() or "MSE" in captured.out

    def test_output_file_creates_json(self, tmp_path, capsys):
        """--output flag creates JSON results file."""
        import json

        from opifex.benchmarking.cli import run_cli

        output_file = tmp_path / "results.json"

        exit_code = run_cli(
            [
                "-b",
                "PDEBench_2D_DarcyFlow",
                "-o",
                "TensorizedFourierNeuralOperator",
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--output",
                str(output_file),
            ]
        )

        assert exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            result = json.load(f)

        assert "model_name" in result
        assert "metrics" in result
        assert result["model_name"] == "TensorizedFourierNeuralOperator"


class TestCLIModuleExecution:
    """Tests for running CLI as module."""

    def test_module_executable(self):
        """CLI can be run as python -m opifex.benchmarking.cli."""
        import subprocess
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent

        result = subprocess.run(
            ["python", "-m", "opifex.benchmarking.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        assert result.returncode == 0
        assert "benchmark" in result.stdout.lower()
