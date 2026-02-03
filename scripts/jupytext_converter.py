#!/usr/bin/env python
r"""Jupytext Conversion & Synchronization Utility

A comprehensive tool for converting and synchronizing between Python scripts (.py)
and Jupyter notebooks (.ipynb) using Jupytext's py:percent format.

Features:
    - Bidirectional conversion: .py ↔ .ipynb
    - Pre-conversion validation to detect common issues
    - Batch conversion for directories
    - Format pairing and synchronization
    - Deterministic output (removes cell IDs)

Usage:
    # Convert single file
    python scripts/jupytext_converter.py py-to-nb examples/path/to/example.py
    python scripts/jupytext_converter.py nb-to-py examples/path/to/example.ipynb

    # Sync existing pair (bidirectional)
    python scripts/jupytext_converter.py sync examples/path/to/example.py

    # Batch convert directory
    python scripts/jupytext_converter.py batch-py-to-nb examples/generative_models/
    python scripts/jupytext_converter.py batch-nb-to-py examples/generative_models/

    # Validate synchronization
    python scripts/jupytext_converter.py validate examples/

Requirements:
    - jupytext: Install with `uv add jupytext` or `pip install jupytext`

Author: Artifex Team
Last Updated: 2025-10-16

IMPORTANT - Known Jupytext Limitations:
==========================================

1. **String Literals with Escape Sequences**:
   Jupytext's py:percent format interprets escape sequences like "\n" as actual
   newlines when converting to .ipynb format. This causes string concatenation
   to be split across multiple lines in the notebook's source array, creating
   invalid Python syntax that fails linting.

   Problem:
       print("\n" + "=" * 80)  # Gets split in notebook as: ['print("\n', '" + "=" * 80)']

   Solutions (in order of preference):
       1. Use separate statements (most Pythonic):
          print()
          print("=" * 80)

       2. Use chr() function:
          print(chr(10) + "=" * 80)

       3. Use multiline string:
          print('''
''' + "=" * 80)

   This is a jupytext parsing behavior, not a configuration issue. There are no
   format options to prevent this splitting. Always write code that avoids
   escape sequences in string concatenation when possible.

   Reference: Investigated 2025-10-16 during vae_mnist.py refactoring.
   See: https://github.com/mwouts/jupytext (no specific issue filed for this)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def validate_python_for_jupytext(py_file: Path) -> tuple[bool, list[str]]:
    r"""Validate Python file for common jupytext conversion issues.

    Checks for patterns that will cause problems when converting to notebooks:
    1. String literals with escape sequences (e.g., "\n") in concatenation
    2. Other known problematic patterns

    Args:
        py_file: Path to Python file to validate

    Returns:
        tuple of (is_valid: bool, issues: list[str])
    """
    issues = []

    try:
        with open(py_file, encoding="utf-8") as f:
            lines = f.readlines()

        # Pattern 1: print() with "\n" or other escape sequences in string concatenation
        # Matches: print("\n" + ...) or print(... + "\n") etc.
        escape_concat_pattern = re.compile(
            r'print\s*\(\s*["\']\\[ntr]["\']\s*[\+]|'  # print("\n" +
            r'[\+]\s*["\']\\[ntr]["\']\s*\)'  # + "\n")
        )

        for line_num, line in enumerate(lines, 1):
            # Check for escape sequence concatenation in print statements
            if escape_concat_pattern.search(line):
                issues.append(
                    f"Line {line_num}: String concatenation with escape sequences in print()\n"
                    f"  Found: {line.strip()}\n"
                    f"  Issue: Jupytext splits escape sequences at actual newlines, "
                    f"causing syntax errors\n"
                    f"  Fix: Use separate print() statements:\n"
                    f'    Before: print("\\n" + "=" * 80)\n'
                    f"    After:  print()\n"
                    f'            print("=" * 80)'
                )

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Error reading file: {e}")
        return False, issues


def run_jupytext_command(args: list[str], verbose: bool = False) -> tuple[bool, str]:
    """Run a jupytext command.

    Args:
        args: Command arguments to pass to jupytext
        verbose: If True, print command output

    Returns:
        tuple of (success: bool, output: str)
    """
    cmd = ["jupytext", *args]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )

        if verbose or result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out: {' '.join(cmd)}", file=sys.stderr)
        return False, ""
    except Exception as e:
        print(f"❌ Error running jupytext: {e}", file=sys.stderr)
        return False, ""


def convert_py_to_nb(py_file: Path, verbose: bool = False) -> bool:
    """Convert Python script to Jupyter notebook.

    Args:
        py_file: Path to .py file
        verbose: If True, show detailed output

    Returns:
        True if conversion succeeded
    """
    if not py_file.exists():
        print(f"❌ File not found: {py_file}")
        return False

    if py_file.suffix != ".py":
        print(f"❌ Not a Python file: {py_file}")
        return False

    # Pre-conversion validation
    is_valid, issues = validate_python_for_jupytext(py_file)
    if not is_valid:
        msg = (
            f"⚠️  Warning: Found {len(issues)} potential issue(s) "
            "that may cause conversion problems:"
        )
        print(msg)
        print()
        for issue in issues:
            print(f"  {issue}")
            print()
        print("  These patterns will cause jupytext to create invalid notebook syntax.")
        print(
            "  Please fix these issues before converting, or the notebook will fail linting."
        )
        return False

    nb_file = py_file.with_suffix(".ipynb")

    print(f"Converting {py_file} → {nb_file}")

    # Use percent format for Python files
    # Use --update to avoid overwriting cell IDs if notebook exists
    # Use --set-formats to establish pairing
    success, _ = run_jupytext_command(
        [
            "--to",
            "ipynb",
            "--set-formats",
            "py:percent,ipynb",
            "--update-metadata",
            '{"jupytext":{"cell_metadata_filter":"-all"}}',
            str(py_file),
        ],
        verbose,
    )

    if success:
        # Post-process to remove cell IDs for deterministic output
        import json

        try:
            with open(nb_file) as f:
                notebook = json.load(f)

            # Remove cell IDs
            for cell in notebook.get("cells", []):
                cell.pop("id", None)

            # Write back
            with open(nb_file) as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                f.write("\n")  # Add trailing newline

            print(f"✅ Created {nb_file}")
            return True
        except Exception as e:
            print(f"⚠️  Created {nb_file} but failed to strip cell IDs: {e}")
            return True  # Still consider it success

    print(f"❌ Failed to convert {py_file}")
    return False


def convert_nb_to_py(nb_file: Path, verbose: bool = False) -> bool:
    """Convert Jupyter notebook to Python script.

    Args:
        nb_file: Path to .ipynb file
        verbose: If True, show detailed output

    Returns:
        True if conversion succeeded
    """
    if not nb_file.exists():
        print(f"❌ File not found: {nb_file}")
        return False

    if nb_file.suffix != ".ipynb":
        print(f"❌ Not a notebook file: {nb_file}")
        return False

    py_file = nb_file.with_suffix(".py")

    print(f"Converting {nb_file} → {py_file}")

    # Use percent format for Python output
    success, _ = run_jupytext_command(
        ["--to", "py:percent", "--set-formats", "ipynb,py:percent", str(nb_file)],
        verbose,
    )

    if success:
        print(f"✅ Created {py_file}")
        return True

    print(f"❌ Failed to convert {nb_file}")
    return False


def sync_pair(file_path: Path, verbose: bool = False) -> bool:
    """Synchronize a .py and .ipynb pair.

    Args:
        file_path: Path to either .py or .ipynb file
        verbose: If True, show detailed output

    Returns:
        True if sync succeeded
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Determine both file paths
    if file_path.suffix == ".py":
        py_file = file_path
        nb_file = file_path.with_suffix(".ipynb")
    elif file_path.suffix == ".ipynb":
        nb_file = file_path
        py_file = file_path.with_suffix(".py")
    else:
        print(f"❌ Invalid file type: {file_path}")
        return False

    # Check if pair exists
    if not py_file.exists():
        print(f"⚠️  Python file not found: {py_file}")
        print(f"   Creating from {nb_file}...")
        return convert_nb_to_py(nb_file, verbose)

    if not nb_file.exists():
        print(f"⚠️  Notebook not found: {nb_file}")
        print(f"   Creating from {py_file}...")
        return convert_py_to_nb(py_file, verbose)

    print(f"Syncing {py_file} ↔ {nb_file}")

    # Use jupytext --sync to synchronize both files
    success, _ = run_jupytext_command(
        ["--sync", "--set-formats", "py:percent,ipynb", str(py_file)], verbose
    )

    if success:
        print("✅ Synchronized pair")
        return True

    print("❌ Failed to sync pair")
    return False


def batch_convert_directory(
    directory: Path, source_format: str, verbose: bool = False
) -> tuple[int, int]:
    """Batch convert all files in a directory.

    Args:
        directory: Directory to search
        source_format: Source file extension (e.g., '.py' or '.ipynb')
        verbose: If True, show detailed output

    Returns:
        tuple of (success_count, fail_count)
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return 0, 0

    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return 0, 0

    # Find all source files
    source_files = list(directory.rglob(f"*{source_format}"))

    # Filter out __init__.py and checkpoint files
    source_files = [
        f
        for f in source_files
        if f.name != "__init__.py" and ".ipynb_checkpoints" not in str(f)
    ]

    if not source_files:
        print(f"⚠️  No {source_format} files found in {directory}")
        return 0, 0

    print(f"Found {len(source_files)} file(s) to convert")
    print()

    success_count = 0
    fail_count = 0

    for source_file in source_files:
        if source_format == ".py":
            success = convert_py_to_nb(source_file, verbose)
        elif source_format == ".ipynb":
            success = convert_nb_to_py(source_file, verbose)
        else:
            print(f"❌ Unsupported format: {source_format}")
            fail_count += 1
            continue

        if success:
            success_count += 1
        else:
            fail_count += 1

        print()

    return success_count, fail_count


def validate_sync(directory: Path, verbose: bool = False) -> tuple[int, int, int]:
    """Validate that .py and .ipynb pairs are synchronized.

    Args:
        directory: Directory to check
        verbose: If True, show detailed output

    Returns:
        tuple of (synced_count, out_of_sync_count, missing_pair_count)
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return 0, 0, 0

    # Find all Python files
    py_files = list(directory.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]

    print(f"Validating {len(py_files)} Python file(s)")
    print()

    synced_count = 0
    out_of_sync_count = 0
    missing_pair_count = 0

    for py_file in py_files:
        nb_file = py_file.with_suffix(".ipynb")

        if not nb_file.exists():
            print(f"⚠️  Missing notebook: {py_file} → {nb_file}")
            missing_pair_count += 1
            continue

        # Check if files have jupytext pairing metadata
        # Check the notebook file since that's where --set-formats adds metadata
        try:
            import json

            with open(nb_file) as f:
                notebook = json.load(f)

            # Check if notebook has jupytext metadata with formats
            metadata = notebook.get("metadata", {})
            jupytext_meta = metadata.get("jupytext", {})
            has_formats = "formats" in jupytext_meta

            if has_formats:
                if verbose:
                    print(f"✅ Synced: {py_file}")
                synced_count += 1
            else:
                if verbose:
                    print(f"⚠️  No jupytext pairing metadata: {py_file}")
                # Pair exists but may not have sync metadata
                # Try to set it
                success, _ = run_jupytext_command(
                    ["--set-formats", "py:percent,ipynb", str(py_file)], verbose=False
                )
                if success:
                    if verbose:
                        print(f"✅ Fixed pairing: {py_file}")
                    synced_count += 1
                else:
                    print(f"❌ Out of sync: {py_file}")
                    out_of_sync_count += 1
        except Exception as e:
            print(f"❌ Error checking {py_file}: {e}")
            out_of_sync_count += 1

    return synced_count, out_of_sync_count, missing_pair_count


def main():  # noqa: PLR0915
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Jupytext Conversion & Synchronization Utility",
        epilog="For more information, see the docstring at the top of this file.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # py-to-nb command
    py_to_nb = subparsers.add_parser(
        "py-to-nb", help="Convert Python script to notebook"
    )
    py_to_nb.add_argument("file", type=Path, help="Python file to convert")
    py_to_nb.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # nb-to-py command
    nb_to_py = subparsers.add_parser(
        "nb-to-py", help="Convert notebook to Python script"
    )
    nb_to_py.add_argument("file", type=Path, help="Notebook file to convert")
    nb_to_py.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # sync command
    sync = subparsers.add_parser("sync", help="Synchronize .py and .ipynb pair")
    sync.add_argument("file", type=Path, help="Either .py or .ipynb file")
    sync.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # batch-py-to-nb command
    batch_py_to_nb = subparsers.add_parser(
        "batch-py-to-nb", help="Batch convert Python scripts to notebooks"
    )
    batch_py_to_nb.add_argument(
        "directory", type=Path, help="Directory to search for .py files"
    )
    batch_py_to_nb.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # batch-nb-to-py command
    batch_nb_to_py = subparsers.add_parser(
        "batch-nb-to-py", help="Batch convert notebooks to Python scripts"
    )
    batch_nb_to_py.add_argument(
        "directory", type=Path, help="Directory to search for .ipynb files"
    )
    batch_nb_to_py.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # validate command
    validate = subparsers.add_parser(
        "validate", help="Validate synchronization of .py and .ipynb pairs"
    )
    validate.add_argument("directory", type=Path, help="Directory to check")
    validate.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print("=" * 80)
    print(f"Jupytext Converter: {args.command}")
    print("=" * 80)
    print()

    # Execute command
    if args.command == "py-to-nb":
        success = convert_py_to_nb(args.file, args.verbose)
        sys.exit(0 if success else 1)

    elif args.command == "nb-to-py":
        success = convert_nb_to_py(args.file, args.verbose)
        sys.exit(0 if success else 1)

    elif args.command == "sync":
        success = sync_pair(args.file, args.verbose)
        sys.exit(0 if success else 1)

    elif args.command == "batch-py-to-nb":
        success_count, fail_count = batch_convert_directory(
            args.directory, ".py", args.verbose
        )
        print("=" * 80)
        print(f"✅ Succeeded: {success_count}")
        print(f"❌ Failed: {fail_count}")
        sys.exit(0 if fail_count == 0 else 1)

    elif args.command == "batch-nb-to-py":
        success_count, fail_count = batch_convert_directory(
            args.directory, ".ipynb", args.verbose
        )
        print("=" * 80)
        print(f"✅ Succeeded: {success_count}")
        print(f"❌ Failed: {fail_count}")
        sys.exit(0 if fail_count == 0 else 1)

    elif args.command == "validate":
        synced, out_of_sync, missing = validate_sync(args.directory, args.verbose)
        print()
        print("=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print(f"✅ Synced: {synced}")
        print(f"❌ Out of sync: {out_of_sync}")
        print(f"⚠️  Missing pairs: {missing}")
        sys.exit(0 if (out_of_sync == 0 and missing == 0) else 1)


if __name__ == "__main__":
    main()
