#!/usr/bin/env python
"""
JAX Ecosystem Compatibility Checker for Opifex Framework.

This script checks for compatibility between JAX, Flax, and other
Opifex dependencies based on known version constraints.

Usage:
    python scripts/check_opifex_compatibility.py
    python scripts/check_opifex_compatibility.py --verbose
    python scripts/check_opifex_compatibility.py --fix
"""

import argparse
import importlib.metadata
import subprocess
import sys

from packaging import version


def get_version(package_name: str) -> str | None:
    """Get the version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_opifex_compatibility() -> tuple[bool, dict[str, str | None], list[str]]:
    """Check if installed package versions are compatible with Opifex requirements.

    Returns:
        Tuple of (is_compatible, versions_dict, error_messages)
    """
    # Core Opifex dependencies
    versions = {
        "jax": get_version("jax"),
        "flax": get_version("flax"),
        "jaxlib": get_version("jaxlib"),
        "optax": get_version("optax"),
        "orbax-checkpoint": get_version("orbax-checkpoint"),
        "diffrax": get_version("diffrax"),
        "optimistix": get_version("optimistix"),
        "lineax": get_version("lineax"),
        "blackjax": get_version("blackjax"),
        "distrax": get_version("distrax"),
        "beartype": get_version("beartype"),
        "jaxtyping": get_version("jaxtyping"),
        "matplotlib": get_version("matplotlib"),
        "numpy": get_version("numpy"),
    }

    errors = []

    # If core packages are not installed, they can't be compatible
    if not versions["jax"]:
        errors.append("JAX is not installed - this is required for Opifex")
        return False, versions, errors

    if not versions["flax"]:
        errors.append("Flax is not installed - this is required for Opifex")
        return False, versions, errors

    # Parse versions for detailed checking
    jax_v = version.parse(versions["jax"])
    flax_v = version.parse(versions["flax"])

    # Opifex specific requirements
    min_jax_version = version.parse("0.6.1")
    min_flax_version = version.parse("0.10.6")

    # Check minimum versions for Opifex
    if jax_v < min_jax_version:
        errors.append(
            f"JAX {jax_v} is too old. Opifex requires JAX >= {min_jax_version}"
        )

    if flax_v < min_flax_version:
        errors.append(
            f"Flax {flax_v} is too old. Opifex requires Flax >= {min_flax_version}"
        )

    # Check if jaxlib version matches jax version (critical for GPU support)
    if versions["jaxlib"]:
        jaxlib_v = version.parse(versions["jaxlib"])
        # Allow minor version differences but major version should match
        if jaxlib_v.major != jax_v.major or abs(jaxlib_v.minor - jax_v.minor) > 1:
            errors.append(
                f"JAXlib {jaxlib_v} incompatible with JAX {jax_v}. "
                f"Should be same major version."
            )

    # Check Flax NNX compatibility (requires Flax >= 0.10.0)
    if flax_v < version.parse("0.10.0"):
        errors.append(
            "Flax NNX requires Flax >= 0.10.0. Opifex uses FLAX NNX extensively."
        )

    # Check JAX-Flax compatibility
    if flax_v >= version.parse("0.10.0") and jax_v < version.parse("0.5.1"):
        errors.append(
            f"Flax {flax_v} requires JAX >= 0.5.1, but JAX {jax_v} is installed"
        )

    # Check optax compatibility
    if versions["optax"]:
        optax_v = version.parse(versions["optax"])
        if optax_v < version.parse("0.2.0"):
            errors.append(
                f"Optax {optax_v} is too old for modern JAX. Recommend >= 0.2.0"
            )
        elif optax_v >= version.parse("0.2.0") and jax_v < version.parse("0.5.1"):
            errors.append(f"Optax {optax_v} requires JAX >= 0.5.1")

    # Check Orbax compatibility (important for checkpointing)
    if versions["orbax-checkpoint"]:
        orbax_v = version.parse(versions["orbax-checkpoint"])
        if orbax_v >= version.parse("0.11.0") and jax_v < version.parse("0.4.34"):
            errors.append(f"Orbax {orbax_v} requires JAX >= 0.4.34")

    # Check scientific computing packages
    scientific_packages = {
        "diffrax": ("0.7.0", "ODE/SDE solving"),
        "optimistix": ("0.0.10", "Root finding and optimization"),
        "lineax": ("0.0.8", "Linear algebra operations"),
        "blackjax": ("1.2.0", "MCMC sampling"),
        "distrax": ("0.1.5", "Probabilistic distributions"),
    }

    for pkg, (min_ver, description) in scientific_packages.items():
        if versions[pkg]:  # Check that package is installed (not None)
            pkg_version = versions[pkg]
            assert (
                pkg_version is not None
            )  # Help type checker understand this is not None
            pkg_v = version.parse(pkg_version)
            min_v = version.parse(min_ver)
            if pkg_v < min_v:
                errors.append(
                    f"{pkg} {pkg_v} is too old. "
                    f"Opifex recommends >= {min_v} for {description}"
                )

    # Check type checking packages
    if versions["beartype"] and versions["jaxtyping"]:
        beartype_v = version.parse(versions["beartype"])
        if beartype_v < version.parse("0.18.0"):
            errors.append(
                "Beartype < 0.18.0 may have compatibility issues with JAX arrays"
            )

    # Check NumPy compatibility (JAX has specific requirements)
    if versions["numpy"]:
        numpy_v = version.parse(versions["numpy"])
        if numpy_v >= version.parse("2.0.0"):
            errors.append(
                "NumPy 2.0+ may have compatibility issues with JAX. "
                "Consider NumPy < 2.0"
            )

    return len(errors) == 0, versions, errors


def get_opifex_recommended_versions() -> dict[str, str]:
    """Get the recommended versions for Opifex development."""
    return {
        "jax": "0.6.1",
        "flax": "0.10.6",
        "jaxlib": "0.6.1",
        "optax": "0.2.4",
        "orbax-checkpoint": "0.11.13",
        "diffrax": "0.7.0",
        "optimistix": "0.0.10",
        "lineax": "0.0.8",
        "blackjax": "1.2.5",
        "distrax": "0.1.5",
        "beartype": "0.18.5",
        "jaxtyping": "0.2.31",
        "numpy": "1.26.4",
        "matplotlib": "3.8.4",
    }


def generate_installation_commands(
    missing_packages: list[str], outdated_packages: list[str]
) -> list[str]:
    """Generate installation commands to fix compatibility issues."""
    commands = []
    recommended = get_opifex_recommended_versions()

    if missing_packages or outdated_packages:
        # Core JAX ecosystem
        core_packages = ["jax", "flax", "jaxlib", "optax", "orbax-checkpoint"]
        core_updates = [
            pkg
            for pkg in core_packages
            if pkg in missing_packages or pkg in outdated_packages
        ]

        if core_updates:
            # Special handling for JAX with CUDA
            cuda_aware_install = (
                "uv pip install --find-links "
                "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "
                f'"jax[cuda12_pip]>={recommended["jax"]}" '
                f'"jaxlib>={recommended["jaxlib"]}" '
                f'"flax>={recommended["flax"]}" '
                f'"optax>={recommended["optax"]}" '
                f'"orbax-checkpoint>={recommended["orbax-checkpoint"]}"'
            )
            commands.append(
                f"# Core JAX ecosystem with CUDA support:\n{cuda_aware_install}"
            )

            # CPU-only alternative
            cpu_install = (
                f"uv pip install "
                f'"jax>={recommended["jax"]}" '
                f'"jaxlib>={recommended["jaxlib"]}" '
                f'"flax>={recommended["flax"]}" '
                f'"optax>={recommended["optax"]}" '
                f'"orbax-checkpoint>={recommended["orbax-checkpoint"]}"'
            )
            commands.append(f"# CPU-only alternative:\n{cpu_install}")

        # Scientific computing packages
        scientific_packages = ["diffrax", "optimistix", "lineax", "blackjax", "distrax"]
        scientific_updates = [
            pkg
            for pkg in scientific_packages
            if pkg in missing_packages or pkg in outdated_packages
        ]

        if scientific_updates:
            scientific_install = " ".join(
                [f'"{pkg}>={recommended[pkg]}"' for pkg in scientific_updates]
            )
            commands.append(
                f"# Scientific computing packages:\nuv pip install {scientific_install}"
            )

        # Type checking packages
        type_packages = ["beartype", "jaxtyping"]
        type_updates = [
            pkg
            for pkg in type_packages
            if pkg in missing_packages or pkg in outdated_packages
        ]

        if type_updates:
            type_install = " ".join(
                [f'"{pkg}>={recommended[pkg]}"' for pkg in type_updates]
            )
            commands.append(f"# Type checking packages:\nuv pip install {type_install}")

    return commands


def print_detailed_report(
    versions: dict[str, str | None], errors: list[str], verbose: bool = False
) -> None:
    """Print detailed compatibility report."""
    print("📋 Opifex JAX Ecosystem Compatibility Report")
    print("=" * 60)

    # Installed versions
    print("\n📦 Installed Versions:")
    recommended = get_opifex_recommended_versions()

    for pkg, installed_ver in versions.items():
        if installed_ver:
            recommended_ver = recommended.get(pkg, "Not specified")
            if recommended_ver != "Not specified":
                installed_v = version.parse(installed_ver)
                recommended_v = version.parse(recommended_ver)

                if installed_v >= recommended_v:
                    status = "✅"
                elif installed_v.major == recommended_v.major:
                    status = "⚠️"
                else:
                    status = "❌"
            else:
                status = "i"

            print(f"  {status} {pkg}: {installed_ver} (recommended: {recommended_ver})")
        else:
            print(
                f"  ❌ {pkg}: Not installed "
                f"(recommended: {recommended.get(pkg, 'Not specified')})"
            )

    # Compatibility status
    print("\n🔍 Compatibility Status:")
    if not errors:
        print("✅ All packages are compatible with Opifex requirements!")
    else:
        print(f"❌ Found {len(errors)} compatibility issue(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    # Missing and outdated packages
    missing_packages = [
        pkg for pkg, ver in versions.items() if ver is None and pkg in recommended
    ]
    outdated_packages = []

    for pkg, installed_ver in versions.items():
        if installed_ver and pkg in recommended:
            if version.parse(installed_ver) < version.parse(recommended[pkg]):
                outdated_packages.append(pkg)

    if missing_packages:
        print(f"\n📥 Missing Packages: {', '.join(missing_packages)}")

    if outdated_packages:
        print(f"\n📤 Outdated Packages: {', '.join(outdated_packages)}")

    # Installation commands
    commands = generate_installation_commands(missing_packages, outdated_packages)
    if commands:
        print("\n🛠️  Installation Commands:")
        for cmd in commands:
            print(f"\n{cmd}")

    # Additional recommendations
    if verbose:
        print("\n💡 Additional Recommendations:")
        print("1. Use 'uv' package manager for faster dependency resolution")
        print("2. Create a virtual environment for Opifex development")
        print("3. Run 'python scripts/verify_opifex_gpu.py' after installation")
        print("4. Check 'python scripts/gpu_utils.py --comprehensive' for GPU setup")


def fix_compatibility_issues() -> bool:
    """Attempt to automatically fix compatibility issues."""
    print("🔧 Attempting to fix Opifex compatibility issues...")

    _, versions, errors = check_opifex_compatibility()

    if not errors:
        print("✅ No compatibility issues found!")
        return True

    # Get missing and outdated packages
    recommended = get_opifex_recommended_versions()
    missing_packages = [
        pkg for pkg, ver in versions.items() if ver is None and pkg in recommended
    ]
    outdated_packages = []

    for pkg, installed_ver in versions.items():
        if installed_ver and pkg in recommended:
            if version.parse(installed_ver) < version.parse(recommended[pkg]):
                outdated_packages.append(pkg)

    commands = generate_installation_commands(missing_packages, outdated_packages)

    if not commands:
        print("⚠️  No automatic fixes available for the detected issues")
        return False

    print("The following commands will be executed:")
    for cmd in commands:
        print(f"\n{cmd}")

    response = input("\nDo you want to proceed? (y/N): ").strip().lower()
    if response != "y":
        print("Aborted by user")
        return False

    # Execute the commands
    success = True
    for cmd in commands:
        if cmd.startswith("#"):
            continue

        print(f"\nExecuting: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("✅ Command executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            success = False

    return success


def main() -> None:
    """Main function to check Opifex compatibility."""
    parser = argparse.ArgumentParser(
        description="Check Opifex JAX ecosystem compatibility"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix compatibility issues"
    )

    args = parser.parse_args()

    print("🧪 Opifex JAX Ecosystem Compatibility Checker")
    print("Based on workshop insights for optimal compatibility")

    if args.fix:
        success = fix_compatibility_issues()
        if success:
            print("\n🎉 Compatibility issues fixed! Re-checking...")
        else:
            print("\n⚠️  Some issues could not be fixed automatically")

    # Run compatibility check
    is_compatible, versions, errors = check_opifex_compatibility()

    # Print detailed report
    print_detailed_report(versions, errors, args.verbose)

    # Summary
    print(f"\n{'=' * 60}")
    if is_compatible:
        print("🎉 SUCCESS: Your environment is compatible with Opifex!")
        print("   You can proceed with Opifex development.")
        print("   Run 'python scripts/verify_opifex_gpu.py' to test GPU setup.")
        sys.exit(0)
    else:
        print("❌ COMPATIBILITY ISSUES DETECTED")
        print(f"   Found {len(errors)} issue(s) that need attention.")
        print("   Use --fix to attempt automatic resolution.")
        print("   Check project documentation for detailed dependency guidance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
