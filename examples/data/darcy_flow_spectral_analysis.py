#!/usr/bin/env python3
"""Comprehensive Spectral Analysis for Darcy Flow Problems.

This module provides spectral analysis tools for Darcy flow datasets,
including power spectral density, frequency domain analysis, and
spectral visualization capabilities for neural operator validation.

This example reproduces the functionality of `plot_darcy_flow_spectrum.py` from the neuraloperator
repository using the Opifex framework. It demonstrates:

- Spectral analysis using jax.scipy.fft
- Frequency domain visualization
- Modal analysis of solution fields
- Power spectral density computation
- Multi-resolution spectral comparison

Based on: neuraloperator/examples/data/plot_darcy_flow_spectrum.py
Framework: Opifex JAX/Flax NNX implementation
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Opifex Framework imports
from opifex.data.sources import DarcyDataSource


def compute_power_spectrum_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Compute 2D power spectral density of a field.

    Args:
        field: 2D field array (height, width)

    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    # Compute 2D FFT
    fft_field = jnp.fft.fft2(field)

    # Compute power spectral density
    power_spectrum = jnp.abs(fft_field) ** 2

    # Create frequency arrays
    height, width = field.shape
    freq_y = jnp.fft.fftfreq(height)
    freq_x = jnp.fft.fftfreq(width)

    # Compute radial frequency for averaging
    fy, fx = jnp.meshgrid(freq_y, freq_x, indexing="ij")
    k_radial = jnp.sqrt(fx**2 + fy**2)

    return k_radial, power_spectrum


def radial_average_spectrum(
    k_radial: jax.Array, power_spectrum: jax.Array, n_bins: int = 50
) -> tuple[jax.Array, jax.Array]:
    """
    Compute radially averaged power spectrum.

    Args:
        k_radial: Radial frequency array
        power_spectrum: 2D power spectrum
        n_bins: Number of frequency bins

    Returns:
        Tuple of (frequency_bins, averaged_spectrum)
    """
    # Create frequency bins
    k_max = jnp.max(k_radial)
    k_bins = jnp.linspace(0, k_max, n_bins + 1)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    # Compute average spectrum in each bin
    averaged_spectrum = []
    for i in range(n_bins):
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i + 1])
        avg_power = jnp.mean(power_spectrum[mask]) if jnp.sum(mask) > 0 else 0.0
        averaged_spectrum.append(avg_power)

    return k_centers, jnp.array(averaged_spectrum)


def analyze_spectral_energy_distribution(
    field: jax.Array, cutoff_frequency: float = 0.3
) -> dict[str, float]:
    """
    Analyze energy distribution between low and high frequencies.

    Args:
        field: 2D field array
        cutoff_frequency: Frequency cutoff for low/high separation

    Returns:
        Dictionary with energy distribution statistics
    """
    k_radial, power_spectrum = compute_power_spectrum_2d(field)

    # Separate low and high frequency components
    low_freq_mask = k_radial <= cutoff_frequency
    high_freq_mask = k_radial > cutoff_frequency

    # Compute energy in each band
    total_energy = jnp.sum(power_spectrum)
    low_freq_energy = jnp.sum(power_spectrum[low_freq_mask])
    high_freq_energy = jnp.sum(power_spectrum[high_freq_mask])

    return {
        "total_energy": float(total_energy),
        "low_freq_energy": float(low_freq_energy),
        "high_freq_energy": float(high_freq_energy),
        "low_freq_percentage": float(100 * low_freq_energy / total_energy),
        "high_freq_percentage": float(100 * high_freq_energy / total_energy),
        "cutoff_frequency": cutoff_frequency,
    }


def compute_dominant_modes(field: jax.Array, n_modes: int = 10) -> dict[str, Any]:
    """
    Compute dominant Fourier modes in the field.

    Args:
        field: 2D field array
        n_modes: Number of dominant modes to extract

    Returns:
        Dictionary with dominant mode information
    """
    # Compute 2D FFT
    fft_field = jnp.fft.fft2(field)
    power_spectrum = jnp.abs(fft_field) ** 2

    # Find indices of dominant modes
    flat_power = power_spectrum.flatten()
    dominant_indices = jnp.argpartition(flat_power, -n_modes)[-n_modes:]
    dominant_indices = dominant_indices[jnp.argsort(flat_power[dominant_indices])[::-1]]

    # Convert flat indices to 2D indices
    height, width = field.shape
    mode_y, mode_x = jnp.unravel_index(dominant_indices, (height, width))

    # Compute mode information
    mode_powers = flat_power[dominant_indices]
    mode_frequencies_x = jnp.fft.fftfreq(width)[mode_x]
    mode_frequencies_y = jnp.fft.fftfreq(height)[mode_y]
    mode_amplitudes = jnp.abs(fft_field.flatten()[dominant_indices])
    mode_phases = jnp.angle(fft_field.flatten()[dominant_indices])

    return {
        "mode_indices": (mode_y, mode_x),
        "mode_powers": mode_powers,
        "mode_frequencies": (mode_frequencies_x, mode_frequencies_y),
        "mode_amplitudes": mode_amplitudes,
        "mode_phases": mode_phases,
        "total_power_in_modes": float(jnp.sum(mode_powers)),
        "total_field_power": float(jnp.sum(power_spectrum)),
    }


def analyze_darcy_spectral_properties(
    n_samples: int = 50,
    resolutions: list[int] | None = None,
    viscosity_range: tuple[float, float] = (1e-5, 1e-3),
    key: jax.Array | None = None,
) -> dict[str, Any]:
    """
    Analyze spectral properties of Darcy flow fields across resolutions.

    Args:
        n_samples: Number of samples to analyze
        resolutions: List of grid resolutions to test
        viscosity_range: Range of viscosity values
        key: JAX random key

    Returns:
        Dictionary containing spectral analysis results
    """
    if resolutions is None:
        resolutions = [64, 128]

    if key is None:
        key = jax.random.PRNGKey(42)

    print("🌊 Darcy Flow Spectral Analysis - Opifex Framework")
    print("=" * 60)

    results = {
        "resolutions": resolutions,
        "samples_analyzed": n_samples,
        "spectral_data": {},
        "energy_distributions": {},
        "dominant_modes": {},
    }

    for resolution in resolutions:
        print(f"\n📊 Analyzing spectral properties at {resolution}x{resolution}")

        # Create data source (Grain-based)
        data_source = DarcyDataSource(
            n_samples=min(n_samples, 100),
            resolution=resolution,
            viscosity_range=viscosity_range,
            seed=int(key[0]),
        )

        # Generate data
        start_time = time.time()
        samples = [data_source[i] for i in range(len(data_source))]
        train_inputs = jnp.stack([s["input"] for s in samples])
        train_outputs = jnp.stack([s["output"] for s in samples])
        generation_time = time.time() - start_time

        print(f"  ⏱️  Generation time: {generation_time:.3f}s")

        # Analyze multiple samples for statistics
        input_spectra = []
        output_spectra = []
        energy_distributions = []
        dominant_modes_data = []

        n_analyze = min(10, len(train_inputs))  # Analyze first 10 samples
        print(f"  🔍 Analyzing {n_analyze} samples for spectral properties...")

        for i in range(n_analyze):
            input_field = train_inputs[i, 0]  # Remove channel dimension
            output_field = train_outputs[i, 0]

            # Compute power spectra
            k_radial_in, power_in = compute_power_spectrum_2d(input_field)
            k_radial_out, power_out = compute_power_spectrum_2d(output_field)

            # Radially averaged spectra
            _k_centers_in, avg_spectrum_in = radial_average_spectrum(
                k_radial_in, power_in
            )
            k_centers_out, avg_spectrum_out = radial_average_spectrum(
                k_radial_out, power_out
            )

            input_spectra.append(avg_spectrum_in)
            output_spectra.append(avg_spectrum_out)

            # Energy distribution analysis
            energy_dist = analyze_spectral_energy_distribution(output_field)
            energy_distributions.append(energy_dist)

            # Dominant modes analysis
            modes = compute_dominant_modes(output_field)
            dominant_modes_data.append(modes)

        # Compute average spectra across samples
        avg_input_spectrum = jnp.mean(jnp.array(input_spectra), axis=0)
        avg_output_spectrum = jnp.mean(jnp.array(output_spectra), axis=0)
        std_input_spectrum = jnp.std(jnp.array(input_spectra), axis=0)
        std_output_spectrum = jnp.std(jnp.array(output_spectra), axis=0)

        # Average energy distributions
        avg_energy_dist = {
            "low_freq_percentage": np.mean(
                [ed["low_freq_percentage"] for ed in energy_distributions]
            ),
            "high_freq_percentage": np.mean(
                [ed["high_freq_percentage"] for ed in energy_distributions]
            ),
            "total_energy": np.mean(
                [ed["total_energy"] for ed in energy_distributions]
            ),
        }

        print(
            f"  📈 Low frequency energy: {avg_energy_dist['low_freq_percentage']:.1f}%"
        )
        print(
            f"  📉 High frequency energy: {avg_energy_dist['high_freq_percentage']:.1f}%"
        )

        # Store results
        results["spectral_data"][resolution] = {
            "k_centers": k_centers_out,
            "avg_input_spectrum": avg_input_spectrum,
            "avg_output_spectrum": avg_output_spectrum,
            "std_input_spectrum": std_input_spectrum,
            "std_output_spectrum": std_output_spectrum,
            "generation_time": generation_time,
        }

        results["energy_distributions"][resolution] = avg_energy_dist
        results["dominant_modes"][resolution] = dominant_modes_data[
            0
        ]  # Store first sample's modes

    return results


def _plot_power_spectra_comparison(axes, results, colors):
    """Plot power spectra comparison across resolutions."""
    resolutions = results["resolutions"]
    for i, resolution in enumerate(resolutions):
        spectral_data = results["spectral_data"][resolution]
        k_centers = spectral_data["k_centers"]
        avg_spectrum = spectral_data["avg_output_spectrum"]
        std_spectrum = spectral_data["std_output_spectrum"]

        color = colors[i % len(colors)]
        axes.loglog(
            k_centers,
            avg_spectrum,
            label=f"{resolution}x{resolution}",
            color=color,
            linewidth=2,
        )
        axes.fill_between(
            k_centers,
            avg_spectrum - std_spectrum,
            avg_spectrum + std_spectrum,
            alpha=0.3,
            color=color,
        )

    axes.set_xlabel("Frequency")
    axes.set_ylabel("Power Spectral Density")
    axes.set_title("Output Field Power Spectra")
    axes.legend()
    axes.grid(True, alpha=0.3)


def _plot_input_vs_output_spectra(axes, results):
    """Plot input vs output spectra for highest resolution."""
    resolutions = results["resolutions"]
    highest_res = max(resolutions)
    spectral_data = results["spectral_data"][highest_res]
    k_centers = spectral_data["k_centers"]

    axes.loglog(
        k_centers,
        spectral_data["avg_input_spectrum"],
        label="Input (Permeability)",
        linewidth=2,
        color="blue",
    )
    axes.loglog(
        k_centers,
        spectral_data["avg_output_spectrum"],
        label="Output (Pressure)",
        linewidth=2,
        color="red",
    )
    axes.set_xlabel("Frequency")
    axes.set_ylabel("Power Spectral Density")
    axes.set_title(f"Input vs Output Spectra ({highest_res}x{highest_res})")
    axes.legend()
    axes.grid(True, alpha=0.3)


def _plot_energy_distribution(axes, results):
    """Plot energy distribution by frequency band."""
    resolutions = results["resolutions"]
    low_freq_percentages = [
        results["energy_distributions"][r]["low_freq_percentage"] for r in resolutions
    ]
    high_freq_percentages = [
        results["energy_distributions"][r]["high_freq_percentage"] for r in resolutions
    ]

    x = np.arange(len(resolutions))
    width = 0.35

    axes.bar(
        x - width / 2,
        low_freq_percentages,
        width,
        label="Low Frequency",
        color="blue",
        alpha=0.7,
    )
    axes.bar(
        x + width / 2,
        high_freq_percentages,
        width,
        label="High Frequency",
        color="red",
        alpha=0.7,
    )

    axes.set_xlabel("Resolution")
    axes.set_ylabel("Energy Percentage (%)")
    axes.set_title("Energy Distribution by Frequency Band")
    axes.set_xticks(x)
    axes.set_xticklabels([f"{r}x{r}" for r in resolutions])
    axes.legend()
    axes.grid(True, alpha=0.3)


def _plot_spectral_slopes(axes, results, colors):
    """Plot spectral slope analysis."""
    resolutions = results["resolutions"]
    for i, resolution in enumerate(resolutions):
        spectral_data = results["spectral_data"][resolution]
        k_centers = spectral_data["k_centers"]
        avg_spectrum = spectral_data["avg_output_spectrum"]

        # Fit power law to high frequency part
        mask = k_centers > 0.1  # High frequency region
        if jnp.sum(mask) > 5:  # Ensure enough points for fitting
            log_k = jnp.log(k_centers[mask])
            log_spectrum = jnp.log(avg_spectrum[mask])

            # Simple linear regression for slope
            A = jnp.vstack([log_k, jnp.ones(len(log_k))]).T
            slope, intercept = jnp.linalg.lstsq(A, log_spectrum, rcond=None)[0]

            color = colors[i % len(colors)]
            axes.plot(
                k_centers[mask],
                jnp.exp(intercept) * k_centers[mask] ** slope,
                "--",
                color=color,
                alpha=0.7,
                label=f"{resolution}x{resolution}: slope={slope:.2f}",
            )
            axes.loglog(k_centers, avg_spectrum, color=color, linewidth=2)

    axes.set_xlabel("Frequency")
    axes.set_ylabel("Power Spectral Density")
    axes.set_title("Spectral Slopes (High Frequency)")
    axes.legend()
    axes.grid(True, alpha=0.3)


def _plot_dominant_modes(axes, results):
    """Plot dominant modes visualization."""
    resolutions = results["resolutions"]
    highest_res = max(resolutions)
    modes_data = results["dominant_modes"][highest_res]
    mode_powers = modes_data["mode_powers"]
    mode_y, mode_x = modes_data["mode_indices"]

    # Create mode locations plot
    scatter = axes.scatter(
        mode_x, mode_y, c=mode_powers, s=100, cmap="viridis", alpha=0.8
    )
    axes.set_xlabel("Mode X Index")
    axes.set_ylabel("Mode Y Index")
    axes.set_title(f"Dominant Modes Location ({highest_res}x{highest_res})")
    plt.colorbar(scatter, ax=axes, label="Mode Power")
    axes.grid(True, alpha=0.3)


def _plot_generation_performance(axes, results):
    """Plot generation time vs resolution."""
    resolutions = results["resolutions"]
    generation_times = [
        results["spectral_data"][r]["generation_time"] for r in resolutions
    ]

    axes.plot(resolutions, generation_times, "o-", linewidth=2, markersize=8)
    axes.set_xlabel("Resolution")
    axes.set_ylabel("Generation Time (s)")
    axes.set_title("Dataset Generation Performance")
    axes.grid(True, alpha=0.3)


def create_spectral_visualization(
    results: dict[str, Any], save_path: str | None = None
) -> None:
    """
    Create comprehensive spectral analysis visualization.

    Args:
        results: Spectral analysis results
        save_path: Optional path to save the figure
    """
    print("\n📈 Creating Spectral Analysis Visualization")
    print("=" * 45)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Darcy Flow Spectral Analysis - Opifex Framework",
        fontsize=16,
        fontweight="bold",
    )

    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot all sections using helper functions
    _plot_power_spectra_comparison(axes[0, 0], results, colors)
    _plot_input_vs_output_spectra(axes[0, 1], results)
    _plot_energy_distribution(axes[0, 2], results)
    _plot_spectral_slopes(axes[1, 0], results, colors)
    _plot_dominant_modes(axes[1, 1], results)
    _plot_generation_performance(axes[1, 2], results)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  💾 Visualization saved to: {save_path}")

    plt.show()


def main():
    """Main function demonstrating comprehensive Darcy flow spectral analysis."""
    print("🚀 Starting Darcy Flow Spectral Analysis Example")
    print("Using Opifex Framework with JAX/Flax NNX")
    print()

    # Configuration
    config = {
        "n_samples": 50,
        "resolutions": [32, 64, 128],
        "viscosity_range": (1e-5, 1e-3),
    }

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    try:
        # Step 1: Spectral Analysis
        print("Step 1: Spectral Properties Analysis")
        results = analyze_darcy_spectral_properties(
            n_samples=config["n_samples"],
            resolutions=config["resolutions"],
            viscosity_range=config["viscosity_range"],
            key=key,
        )

        # Step 2: Visualization
        print("Step 2: Creating Spectral Visualization")
        create_spectral_visualization(results)

        # Summary
        print("\n✅ Spectral Analysis Complete!")
        print(f"  📊 Analyzed {len(config['resolutions'])} resolutions")
        print("  📈 Computed power spectral densities")
        print("  🔍 Identified dominant modes")
        print("  📉 Analyzed energy distributions")
        print("  📈 Created comprehensive visualization")

        return results

    except Exception as e:
        print(f"❌ Error during spectral analysis: {e}")
        raise


if __name__ == "__main__":
    results = main()
