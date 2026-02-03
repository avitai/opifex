#!/usr/bin/env python3
"""
Enhanced Calibration Methods Demo for Opifex Framework

This example demonstrates the advanced uncertainty calibration capabilities
implemented in the Opifex framework, including:

1. Platt Scaling for binary classification calibration
2. Isotonic Regression for non-parametric calibration
3. Conformal Prediction for coverage guarantees
4. Enhanced Temperature Scaling with physics constraints

These methods provide advanced uncertainty quantification and calibration
for scientific machine learning applications.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian import (
    CalibrationTools,
    ConformalPrediction,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
)


def generate_synthetic_data(key, n_samples=1000):
    """Generate synthetic data for calibration demonstration."""
    # Generate features
    X = jax.random.normal(key, (n_samples, 2))

    # Generate logits with some predictive signal
    true_logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * jnp.sin(X[:, 0])

    # Add noise to create miscalibrated predictions
    noisy_logits = true_logits + 0.3 * jax.random.normal(key, (n_samples,))

    # Generate binary labels based on true logits
    probabilities = jax.nn.sigmoid(true_logits)
    labels = jax.random.bernoulli(key, probabilities)

    # Generate regression targets for conformal prediction demo
    regression_targets = true_logits + 0.2 * jax.random.normal(key, (n_samples,))

    return X, noisy_logits, labels, regression_targets


def demonstrate_platt_scaling():
    """Demonstrate Platt scaling for binary classification calibration."""
    print("\n🔧 PLATT SCALING DEMONSTRATION")
    print("=" * 50)

    # Initialize components
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(42)

    # Generate data
    _X, logits, labels, _ = generate_synthetic_data(key, n_samples=500)

    # Split data for calibration
    train_logits, calib_logits = logits[:300], logits[300:]
    train_labels, calib_labels = labels[:300], labels[300:]

    # Initialize Platt scaling
    platt_scaler = PlattScaling(rngs=rngs)

    print(
        f"Initial parameters: A={platt_scaler.a.value:.3f}, B={platt_scaler.b.value:.3f}"
    )

    # Fit Platt scaling on training data
    platt_scaler.fit(train_logits, train_labels, max_iterations=100)

    print(
        f"Fitted parameters: A={platt_scaler.a.value:.3f}, B={platt_scaler.b.value:.3f}"
    )

    # Apply calibration to validation data
    uncalibrated_probs = jax.nn.sigmoid(calib_logits)
    calibrated_probs = platt_scaler(calib_logits)

    # Compute calibration quality metrics
    def expected_calibration_error(probs, labels, n_bins=10):
        """Compute Expected Calibration Error."""
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = jnp.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = jnp.mean(labels[in_bin])
                avg_confidence_in_bin = jnp.mean(probs[in_bin])
                ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    uncalib_ece = expected_calibration_error(uncalibrated_probs, calib_labels)
    calib_ece = expected_calibration_error(calibrated_probs, calib_labels)

    print(f"Uncalibrated ECE: {uncalib_ece:.4f}")
    print(f"Calibrated ECE: {calib_ece:.4f}")
    print(f"ECE Improvement: {((uncalib_ece - calib_ece) / uncalib_ece * 100):.1f}%")


def demonstrate_isotonic_regression():
    """Demonstrate isotonic regression for non-parametric calibration."""
    print("\n📈 ISOTONIC REGRESSION DEMONSTRATION")
    print("=" * 50)

    # Initialize components
    key = jax.random.PRNGKey(123)
    rngs = nnx.Rngs(123)

    # Generate data with non-linear calibration needs
    _X, logits, labels, _ = generate_synthetic_data(key, n_samples=800)

    # Convert logits to confidences with some distortion
    raw_confidences = jax.nn.sigmoid(logits)
    # Create systematic miscalibration
    distorted_confidences = raw_confidences**1.5  # Over-confident predictions

    # Split data
    train_conf, test_conf = distorted_confidences[:500], distorted_confidences[500:]
    train_labels, _test_labels = labels[:500], labels[500:]

    # Initialize isotonic regression
    isotonic_regressor = IsotonicRegression(n_bins=25, rngs=rngs)

    print(f"Training isotonic regression on {len(train_conf)} samples...")

    # Fit isotonic regression
    isotonic_regressor.fit(train_conf, train_labels)

    # Apply calibration
    calibrated_confidences = isotonic_regressor(test_conf)

    # Compute reliability metrics
    def reliability_diagram_data(confidences, accuracies, n_bins=10):
        """Compute reliability diagram data."""
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_confidences = []
        bin_accuracies = []

        for i in range(n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1]
            )
            if jnp.sum(in_bin) > 0:
                bin_conf = jnp.mean(confidences[in_bin])
                bin_acc = jnp.mean(accuracies[in_bin])
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
            else:
                bin_confidences.append(bin_centers[i])
                bin_accuracies.append(bin_centers[i])

        return jnp.array(bin_confidences), jnp.array(bin_accuracies)

    # Before calibration
    before_conf, before_acc = reliability_diagram_data(test_conf, _test_labels)
    before_reliability = jnp.mean(jnp.abs(before_conf - before_acc))

    # After calibration
    after_conf, after_acc = reliability_diagram_data(
        calibrated_confidences, _test_labels
    )
    after_reliability = jnp.mean(jnp.abs(after_conf - after_acc))

    print(f"Average reliability gap before: {before_reliability:.4f}")
    print(f"Average reliability gap after: {after_reliability:.4f}")
    print(
        f"Reliability improvement: {((before_reliability - after_reliability) / before_reliability * 100):.1f}%"
    )


def demonstrate_conformal_prediction():
    """Demonstrate conformal prediction for coverage guarantees."""
    print("\n🎯 CONFORMAL PREDICTION DEMONSTRATION")
    print("=" * 50)

    # Initialize components
    key = jax.random.PRNGKey(456)
    rngs = nnx.Rngs(456)

    # Generate regression data
    _X, logits, _, targets = generate_synthetic_data(key, n_samples=600)

    # Use logits as predictions (with some error)
    predictions = logits + 0.1 * jax.random.normal(key, logits.shape)

    # Split data for conformal prediction
    calib_pred, test_pred = predictions[:300], predictions[300:]
    calib_targets, test_targets = targets[:300], targets[300:]

    # Test different coverage levels
    coverage_levels = [0.80, 0.90, 0.95]

    print("Testing different coverage levels:")
    print("-" * 30)

    for coverage in coverage_levels:
        alpha = 1 - coverage

        # Initialize conformal predictor
        conformal_predictor = ConformalPrediction(alpha=alpha, rngs=rngs)

        # Calibrate using calibration set
        conformal_predictor.calibrate(calib_pred, calib_targets)

        # Generate prediction intervals for test set
        lower_bounds, upper_bounds = conformal_predictor.predict_intervals(test_pred)

        # Compute empirical coverage
        empirical_coverage = conformal_predictor.compute_coverage(
            lower_bounds, upper_bounds, test_targets
        )

        # Compute average interval width
        avg_width = jnp.mean(upper_bounds - lower_bounds)

        print(f"Target coverage: {coverage:.0%}")
        print(f"Empirical coverage: {empirical_coverage:.3f}")
        print(f"Average interval width: {avg_width:.3f}")
        print(f"Coverage error: {abs(empirical_coverage - coverage):.3f}")
        print()


def demonstrate_enhanced_temperature_scaling():
    """Demonstrate enhanced temperature scaling with adaptive features."""
    print("\n🌡️ ENHANCED TEMPERATURE SCALING DEMONSTRATION")
    print("=" * 50)

    # Initialize components
    key = jax.random.PRNGKey(789)
    rngs = nnx.Rngs(789)

    # Generate data
    X, logits, labels, targets = generate_synthetic_data(key, n_samples=400)

    # Split data
    test_X = X[200:]
    train_logits, test_logits = logits[:200], logits[200:]
    _train_labels, _test_labels = labels[:200], labels[200:]
    _test_targets = targets[200:]

    # Initialize enhanced temperature scaling
    temp_scaler = TemperatureScaling(
        physics_constraints=["energy_conservation"],
        adaptive=True,
        learning_rate=0.02,
        rngs=rngs,
    )

    print(f"Initial temperature: {temp_scaler.temperature.value:.3f}")

    # Optimize temperature
    optimized_temp = temp_scaler.optimize_temperature(
        train_logits, _train_labels.astype(int)
    )

    print(f"Optimized temperature: {optimized_temp:.3f}")

    # Apply temperature scaling to test data
    calibrated_preds, aleatoric_uncertainty = temp_scaler(test_logits[:, None], test_X)
    calibrated_preds = calibrated_preds.squeeze()
    aleatoric_uncertainty = aleatoric_uncertainty.squeeze()

    # Test adaptive temperature scaling
    adaptive_temps = temp_scaler.adaptive_temperature_scaling(
        test_logits, aleatoric_uncertainty, _test_targets
    )

    print(f"Average adaptive temperature: {jnp.mean(adaptive_temps):.3f}")
    print(f"Temperature std: {jnp.std(adaptive_temps):.3f}")
    print(f"Average aleatoric uncertainty: {jnp.mean(aleatoric_uncertainty):.3f}")


def demonstrate_integrated_calibration_pipeline():
    """Demonstrate integrated use of multiple calibration methods."""
    print("\n🔄 INTEGRATED CALIBRATION PIPELINE DEMONSTRATION")
    print("=" * 50)

    # Initialize components
    key = jax.random.PRNGKey(999)
    rngs = nnx.Rngs(999)

    # Generate comprehensive dataset
    X, logits, labels, targets = generate_synthetic_data(key, n_samples=1000)

    # Split data into train/calib/test
    _train_X, _calib_X, test_X = X[:400], X[400:700], X[700:]
    train_logits, calib_logits, test_logits = (
        logits[:400],
        logits[400:700],
        logits[700:],
    )
    train_labels, calib_labels, test_labels = (
        labels[:400],
        labels[400:700],
        labels[700:],
    )
    train_targets, calib_targets, test_targets = (
        targets[:400],
        targets[400:700],
        targets[700:],
    )

    print("Setting up integrated calibration pipeline...")

    # 1. Enhanced CalibrationTools for assessment
    calibration_tools = CalibrationTools(rngs=rngs)

    # Convert logits to predictions for regression assessment
    regression_preds = train_logits + 0.1 * jax.random.normal(key, train_logits.shape)
    uncertainties = jnp.abs(0.2 * jax.random.normal(key, train_logits.shape)) + 0.1

    initial_metrics = calibration_tools.assess_calibration(
        regression_preds, uncertainties, train_targets, num_bins=10
    )

    print(f"Initial ECE: {initial_metrics['expected_calibration_error']:.4f}")
    print(f"Initial MCE: {initial_metrics['maximum_calibration_error']:.4f}")

    # 2. Apply Platt scaling for classification
    platt_scaler = PlattScaling(rngs=rngs)
    platt_scaler.fit(train_logits, train_labels)
    calib_class_probs = platt_scaler(calib_logits)

    # 3. Apply isotonic regression for further refinement
    isotonic_regressor = IsotonicRegression(n_bins=20, rngs=rngs)
    isotonic_regressor.fit(calib_class_probs, calib_labels)
    _refined_probs = isotonic_regressor(calib_class_probs)

    # 4. Apply conformal prediction for regression
    conformal_predictor = ConformalPrediction(alpha=0.1, rngs=rngs)
    conformal_predictor.calibrate(calib_logits, calib_targets)

    # 5. Enhanced temperature scaling
    temp_scaler = TemperatureScaling(
        physics_constraints=["energy_conservation"], adaptive=True, rngs=rngs
    )
    temp_scaler.optimize_temperature(train_logits, train_labels.astype(int))

    # Apply full pipeline to test data
    print("\nApplying full calibration pipeline to test data...")

    # Classification pipeline
    test_class_probs = platt_scaler(test_logits)
    test_refined_probs = isotonic_regressor(test_class_probs)

    # Regression pipeline
    test_lower, test_upper = conformal_predictor.predict_intervals(test_logits)
    _test_calibrated, test_uncertainty = temp_scaler(test_logits[:, None], test_X)

    # Final assessment
    final_coverage = conformal_predictor.compute_coverage(
        test_lower, test_upper, test_targets
    )

    # Classification calibration assessment
    def compute_ece(probs, labels):
        """Simple ECE computation."""
        accuracies = (probs > 0.5) == (labels > 0.5)
        return jnp.mean(jnp.abs(probs - accuracies))

    original_ece = compute_ece(jax.nn.sigmoid(test_logits), test_labels)
    refined_ece = compute_ece(test_refined_probs, test_labels)

    print("\nFinal Results:")
    print(f"Classification ECE improvement: {original_ece:.4f} → {refined_ece:.4f}")
    print(f"Regression coverage: {final_coverage:.3f} (target: 0.900)")
    print(f"Average uncertainty: {jnp.mean(test_uncertainty.squeeze()):.3f}")
    print("Successful integration of all calibration methods! ✅")


def main():
    """Run all calibration demonstrations."""
    print("🎉 Opifex Enhanced Calibration Methods Demonstration")
    print("=" * 60)
    print()
    print("This demo showcases the advanced uncertainty calibration capabilities")
    print("implemented in the Opifex framework, providing advanced")
    print("calibration methods for scientific machine learning applications.")

    try:
        # Run individual method demonstrations
        demonstrate_platt_scaling()
        demonstrate_isotonic_regression()
        demonstrate_conformal_prediction()
        demonstrate_enhanced_temperature_scaling()

        # Run integrated pipeline demonstration
        demonstrate_integrated_calibration_pipeline()

        print("\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key achievements demonstrated:")
        print("✅ Platt Scaling: Parametric binary classification calibration")
        print("✅ Isotonic Regression: Non-parametric monotonic calibration")
        print("✅ Conformal Prediction: Finite-sample coverage guarantees")
        print("✅ Enhanced Temperature Scaling: Adaptive physics-aware calibration")
        print("✅ Integrated Pipeline: Seamless combination of all methods")
        print()
        print("The Opifex framework now provides enterprise-grade uncertainty")
        print("calibration capabilities for scientific computing applications! 🚀")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check the implementation and try again.")


if __name__ == "__main__":
    main()
