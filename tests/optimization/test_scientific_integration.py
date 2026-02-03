"""Tests for scientific computing integration components.

This module tests the Phase 7.4 Scientific Computing Integration implementation
including physics profiling, numerical validation, and conservation checking.
"""

import jax.numpy as jnp
import pytest

from opifex.core.physics import ConservationLaw
from opifex.optimization.scientific_integration import (
    ConservationCheckResult,
    NumericalValidationResult,
    NumericalValidator,
    PhysicsDomain,
    PhysicsMetrics,
    PhysicsProfiler,
    ScientificBenchmarkResult,
    ScientificBenchmarkValidator,
    ScientificComputingIntegrator,
)


@pytest.fixture
def sample_physics_data():
    """Create sample physics data for testing."""
    return {
        "energy": 100.0,
        "momentum": [1.0, 2.0, 3.0],
        "mass": 50.0,
        "total_mass": 50.0,
        "temperature": 300.0,
        "entropy": 25.0,
        "wavefunction": jnp.array([0.6, 0.8]),  # Normalized
        "symmetry_operations": [
            {"type": "rotation", "axis": [0, 0, 1], "angle": jnp.pi / 4},
            {"type": "translation", "vector": [1.0, 0.0, 0.0]},
        ],
        "boundary_conditions": {
            "dirichlet": {
                "indices": [0, 1, 2],
                "values": [0.0, 0.0, 0.0],
            },
            "neumann": {
                "derivative_value": 0.0,
            },
        },
    }


@pytest.fixture
def sample_model_output():
    """Create sample model output for testing."""
    return jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )


@pytest.fixture
def quantum_chemistry_profiler():
    """Create a quantum chemistry physics profiler."""
    return PhysicsProfiler(PhysicsDomain.QUANTUM_CHEMISTRY)


@pytest.fixture
def fluid_dynamics_profiler():
    """Create a fluid dynamics physics profiler."""
    return PhysicsProfiler(PhysicsDomain.FLUID_DYNAMICS)


@pytest.fixture
def numerical_validator():
    """Create a numerical validator."""
    return NumericalValidator(precision_threshold=1e-6, stability_threshold=1e-3)


@pytest.fixture
def benchmark_validator():
    """Create a benchmark validator."""
    return ScientificBenchmarkValidator(PhysicsDomain.QUANTUM_CHEMISTRY)


class TestPhysicsDomain:
    """Test PhysicsDomain enum."""

    def test_physics_domain_values(self):
        """Test all physics domain values."""
        assert PhysicsDomain.QUANTUM_CHEMISTRY.value == "quantum_chemistry"
        assert PhysicsDomain.FLUID_DYNAMICS.value == "fluid_dynamics"
        assert PhysicsDomain.MATERIALS_SCIENCE.value == "materials_science"
        assert PhysicsDomain.PLASMA_PHYSICS.value == "plasma_physics"
        assert PhysicsDomain.MOLECULAR_DYNAMICS.value == "molecular_dynamics"
        assert PhysicsDomain.SOLID_STATE.value == "solid_state"
        assert PhysicsDomain.GENERAL.value == "general"


class TestConservationLaw:
    """Test ConservationLaw enum."""

    def test_conservation_law_values(self):
        """Test all conservation law values."""
        assert ConservationLaw.ENERGY.value == "energy"
        assert ConservationLaw.MOMENTUM.value == "momentum"
        assert ConservationLaw.ANGULAR_MOMENTUM.value == "angular_momentum"
        assert ConservationLaw.MASS.value == "mass"
        assert ConservationLaw.CHARGE.value == "charge"
        assert ConservationLaw.PARTICLE_NUMBER.value == "particle_number"
        assert ConservationLaw.PROBABILITY.value == "probability"


class TestPhysicsMetrics:
    """Test PhysicsMetrics data structure."""

    def test_physics_metrics_creation(self):
        """Test physics metrics creation."""
        metrics = PhysicsMetrics(
            domain=PhysicsDomain.QUANTUM_CHEMISTRY,
            conservation_violations={ConservationLaw.ENERGY: 1e-12},
            symmetry_preservation=0.99,
            numerical_stability=0.95,
            energy_conservation_error=1e-12,
        )

        assert metrics.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert metrics.conservation_violations[ConservationLaw.ENERGY] == 1e-12
        assert metrics.symmetry_preservation == 0.99
        assert metrics.numerical_stability == 0.95
        assert metrics.energy_conservation_error == 1e-12

    def test_physics_metrics_defaults(self):
        """Test physics metrics default values."""
        metrics = PhysicsMetrics(domain=PhysicsDomain.GENERAL)

        assert metrics.domain == PhysicsDomain.GENERAL
        assert isinstance(metrics.conservation_violations, dict)
        assert metrics.symmetry_preservation == 0.0
        assert metrics.numerical_stability == 0.0
        assert metrics.energy_conservation_error == 0.0


class TestPhysicsProfiler:
    """Test PhysicsProfiler component."""

    def test_profiler_initialization_quantum_chemistry(
        self, quantum_chemistry_profiler
    ):
        """Test quantum chemistry profiler initialization."""
        assert quantum_chemistry_profiler.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert "energy_conservation" in quantum_chemistry_profiler.validation_tolerances
        assert (
            quantum_chemistry_profiler.validation_tolerances["energy_conservation"]
            == 1e-12
        )

    def test_profiler_initialization_fluid_dynamics(self, fluid_dynamics_profiler):
        """Test fluid dynamics profiler initialization."""
        assert fluid_dynamics_profiler.domain == PhysicsDomain.FLUID_DYNAMICS
        assert "mass_conservation" in fluid_dynamics_profiler.validation_tolerances
        assert (
            fluid_dynamics_profiler.validation_tolerances["momentum_conservation"]
            == 1e-6
        )

    def test_profile_physics_metrics_basic(
        self, quantum_chemistry_profiler, sample_model_output, sample_physics_data
    ):
        """Test basic physics metrics profiling."""
        metrics = quantum_chemistry_profiler.profile_physics_metrics(
            sample_model_output, sample_physics_data
        )

        assert isinstance(metrics, PhysicsMetrics)
        assert metrics.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert isinstance(metrics.conservation_violations, dict)
        assert 0.0 <= metrics.numerical_stability <= 1.0

    def test_energy_conservation_checking(
        self, quantum_chemistry_profiler, sample_model_output
    ):
        """Test energy conservation checking."""
        reference_data = {"energy": 100.0}

        error = quantum_chemistry_profiler._check_energy_conservation_over_time(
            sample_model_output, reference_data
        )

        assert isinstance(error, float)
        assert error >= 0.0

    def test_momentum_conservation_checking(
        self, fluid_dynamics_profiler, sample_model_output
    ):
        """Test momentum conservation checking."""
        reference_data = {"momentum": jnp.array([1.0, 2.0, 3.0])}

        error = fluid_dynamics_profiler._check_momentum_conservation_over_time(
            sample_model_output, reference_data
        )

        assert isinstance(error, float)
        assert error >= 0.0

    def test_mass_conservation_checking(
        self, fluid_dynamics_profiler, sample_model_output
    ):
        """Test mass conservation checking."""
        reference_data = {"total_mass": 50.0}

        error = fluid_dynamics_profiler._check_mass_conservation_over_time(
            sample_model_output, reference_data
        )

        assert isinstance(error, float)
        assert error >= 0.0

    def test_symmetry_preservation_checking(
        self, quantum_chemistry_profiler, sample_model_output
    ):
        """Test symmetry preservation checking."""
        symmetry_operations = [
            {"type": "rotation", "axis": [0, 0, 1], "angle": jnp.pi / 4},
        ]

        preservation = quantum_chemistry_profiler._check_symmetry_preservation(
            sample_model_output, symmetry_operations
        )

        assert isinstance(preservation, float)
        assert 0.0 <= preservation <= 1.0

    def test_numerical_stability_assessment(
        self, quantum_chemistry_profiler, sample_model_output
    ):
        """Test numerical stability assessment."""
        stability = quantum_chemistry_profiler._assess_numerical_stability(
            sample_model_output
        )

        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    def test_numerical_stability_with_nan(self, quantum_chemistry_profiler):
        """Test numerical stability with NaN values."""
        nan_output = jnp.array([[jnp.nan, 1.0], [2.0, jnp.inf]])

        stability = quantum_chemistry_profiler._assess_numerical_stability(nan_output)

        assert stability == 0.0  # Should detect instability

    def test_unitarity_checking(self, quantum_chemistry_profiler):
        """Test unitarity checking for quantum systems."""
        # Create a unitary matrix (rotation matrix)
        angle = jnp.pi / 4
        unitary_matrix = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )

        reference_data = {"wavefunction": jnp.array([0.6, 0.8])}  # Normalized

        unitarity = quantum_chemistry_profiler._check_unitarity(
            unitary_matrix, reference_data
        )

        assert isinstance(unitarity, float)
        assert 0.0 <= unitarity <= 1.0

    def test_boundary_conditions_checking(self, fluid_dynamics_profiler):
        """Test boundary conditions checking."""
        model_output = jnp.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
        boundary_conditions = {
            "dirichlet": {
                "indices": [0, 1],
                "values": [0.0, 0.0],
            },
        }

        accuracy = fluid_dynamics_profiler._check_boundary_conditions(
            model_output, boundary_conditions
        )

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_thermodynamic_consistency(
        self, quantum_chemistry_profiler, sample_model_output
    ):
        """Test thermodynamic consistency checking."""
        reference_data = {"temperature": 300.0, "entropy": 25.0}

        consistency = quantum_chemistry_profiler._check_thermodynamic_consistency(
            sample_model_output, reference_data
        )

        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0


class TestNumericalValidator:
    """Test NumericalValidator component."""

    def test_validator_initialization(self, numerical_validator):
        """Test numerical validator initialization."""
        assert numerical_validator.precision_threshold == 1e-6
        assert numerical_validator.stability_threshold == 1e-3

    def test_validate_numerical_precision_perfect_match(self, numerical_validator):
        """Test numerical precision validation with perfect match."""
        computed = jnp.array([1.0, 2.0, 3.0])
        reference = jnp.array([1.0, 2.0, 3.0])

        result = numerical_validator.validate_numerical_precision(computed, reference)

        assert isinstance(result, NumericalValidationResult)
        assert result.is_valid
        assert result.precision_score == 1.0
        assert len(result.validation_errors) == 0

    def test_validate_numerical_precision_with_error(self, numerical_validator):
        """Test numerical precision validation with error."""
        computed = jnp.array([1.1, 2.1, 3.1])
        reference = jnp.array([1.0, 2.0, 3.0])

        result = numerical_validator.validate_numerical_precision(computed, reference)

        assert isinstance(result, NumericalValidationResult)
        assert result.precision_score < 1.0
        assert result.condition_number > 0

    def test_validate_numerical_precision_with_nan(self, numerical_validator):
        """Test numerical precision validation with NaN."""
        computed = jnp.array([jnp.nan, 2.0, 3.0])
        reference = jnp.array([1.0, 2.0, 3.0])

        result = numerical_validator.validate_numerical_precision(computed, reference)

        assert isinstance(result, NumericalValidationResult)
        assert not result.is_valid
        assert len(result.validation_errors) > 0
        assert "NaN or Inf values detected" in result.validation_errors[0]

    def test_check_conservation_law_energy(self, numerical_validator):
        """Test energy conservation law checking."""
        computed = jnp.array([100.0])
        reference = jnp.array([100.0])

        result = numerical_validator.check_conservation_law(
            computed, reference, ConservationLaw.ENERGY
        )

        assert isinstance(result, ConservationCheckResult)
        assert result.law == ConservationLaw.ENERGY
        assert result.is_conserved
        assert result.violation_magnitude < 1e-10

    def test_check_conservation_law_violation(self, numerical_validator):
        """Test conservation law with violation."""
        computed = jnp.array([110.0])  # 10% error
        reference = jnp.array([100.0])

        result = numerical_validator.check_conservation_law(
            computed, reference, ConservationLaw.ENERGY, tolerance=1e-6
        )

        assert isinstance(result, ConservationCheckResult)
        assert not result.is_conserved  # Should violate the tolerance
        assert result.violation_magnitude > 1e-6
        assert result.relative_error > 0.01  # 1% relative error


class TestScientificBenchmarkValidator:
    """Test ScientificBenchmarkValidator component."""

    def test_validator_initialization(self, benchmark_validator):
        """Test benchmark validator initialization."""
        assert benchmark_validator.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert "chemical_accuracy" in benchmark_validator.benchmark_thresholds

    def test_validate_benchmark_accurate(self, benchmark_validator):
        """Test benchmark validation with accurate result."""
        result = benchmark_validator.validate_benchmark(
            benchmark_name="water_energy",
            computed_value=-76.4,
            reference_value=-76.4,
            accuracy_type="chemical_accuracy",
        )

        assert isinstance(result, ScientificBenchmarkResult)
        assert result.benchmark_name == "water_energy"
        assert result.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert result.meets_accuracy_threshold
        assert result.accuracy_score == 1.0
        assert result.relative_error == 0.0

    def test_validate_benchmark_inaccurate(self, benchmark_validator):
        """Test benchmark validation with inaccurate result."""
        result = benchmark_validator.validate_benchmark(
            benchmark_name="water_energy",
            computed_value=-75.0,  # 1.4 hartree error (significant)
            reference_value=-76.4,
            accuracy_type="chemical_accuracy",
        )

        assert isinstance(result, ScientificBenchmarkResult)
        assert not result.meets_accuracy_threshold
        assert result.accuracy_score < 1.0
        assert result.relative_error > 0.01

    def test_validate_multiple_benchmarks(self, benchmark_validator):
        """Test validation of multiple benchmarks."""
        benchmarks = {
            "water_energy": (-76.4, -76.4),  # Perfect
            "methane_energy": (-40.5, -40.0),  # Small error
            "benzene_energy": (-230.0, -232.0),  # Larger error
        }

        results = benchmark_validator.validate_multiple_benchmarks(benchmarks)

        assert len(results) == 3
        assert all(isinstance(r, ScientificBenchmarkResult) for r in results)

        # Check that different benchmarks have different accuracy scores
        accuracy_scores = [r.accuracy_score for r in results]
        assert len(set(accuracy_scores)) > 1  # Should have different scores


class TestScientificComputingIntegrator:
    """Test ScientificComputingIntegrator system."""

    def test_integrator_initialization(self):
        """Test integrator initialization."""
        integrator = ScientificComputingIntegrator(PhysicsDomain.QUANTUM_CHEMISTRY)

        assert integrator.domain == PhysicsDomain.QUANTUM_CHEMISTRY
        assert isinstance(integrator.physics_profiler, PhysicsProfiler)
        assert isinstance(integrator.numerical_validator, NumericalValidator)
        assert isinstance(integrator.benchmark_validator, ScientificBenchmarkValidator)

    def test_integrator_custom_components(self):
        """Test integrator with custom components."""
        custom_profiler = PhysicsProfiler(PhysicsDomain.FLUID_DYNAMICS)
        custom_validator = NumericalValidator(precision_threshold=1e-8)
        custom_benchmark = ScientificBenchmarkValidator(PhysicsDomain.FLUID_DYNAMICS)

        integrator = ScientificComputingIntegrator(
            domain=PhysicsDomain.FLUID_DYNAMICS,
            physics_profiler=custom_profiler,
            numerical_validator=custom_validator,
            benchmark_validator=custom_benchmark,
        )

        assert integrator.domain == PhysicsDomain.FLUID_DYNAMICS
        assert integrator.physics_profiler == custom_profiler
        assert integrator.numerical_validator == custom_validator
        assert integrator.benchmark_validator == custom_benchmark

    def test_comprehensive_scientific_validation(self, sample_model_output):
        """Test comprehensive scientific validation."""
        integrator = ScientificComputingIntegrator(PhysicsDomain.QUANTUM_CHEMISTRY)

        reference_data = {
            "physics_reference": {
                "energy": 100.0,
                "momentum": [1.0, 2.0, 3.0],
            },
            "numerical_reference": sample_model_output,
            "energy": jnp.array([100.0]),
            "momentum": jnp.array([1.0, 2.0, 3.0]),
        }

        benchmarks = {
            "test_energy": (100.0, 100.0),
            "test_gradient": (0.5, 0.5),
        }

        results = integrator.comprehensive_scientific_validation(
            sample_model_output, reference_data, benchmarks
        )

        assert isinstance(results, dict)
        assert "domain" in results
        assert "overall_scientific_score" in results
        assert results["domain"] == "quantum_chemistry"
        assert 0.0 <= results["overall_scientific_score"] <= 1.0

    def test_optimize_for_scientific_accuracy(self, sample_model_output):
        """Test optimization recommendations based on validation."""
        integrator = ScientificComputingIntegrator(PhysicsDomain.QUANTUM_CHEMISTRY)

        # Create validation results with some issues
        validation_results = {
            "physics_metrics": PhysicsMetrics(
                domain=PhysicsDomain.QUANTUM_CHEMISTRY,
                numerical_stability=0.8,  # Below 0.9 threshold
                energy_conservation_error=1e-5,  # Above threshold
                symmetry_preservation=0.9,  # Below 0.95 threshold
            ),
            "numerical_validation": NumericalValidationResult(
                is_valid=False,
                precision_score=0.7,
                stability_score=0.8,
                convergence_rate=0.1,
                condition_number=1e8,  # High condition number
                validation_errors=["High numerical error"],
                recommendations=["Use higher precision"],
            ),
            "conservation_checks": [
                ConservationCheckResult(
                    law=ConservationLaw.ENERGY,
                    is_conserved=False,
                    violation_magnitude=1e-5,
                    tolerance=1e-10,
                    relative_error=1e-3,
                ),
            ],
            "benchmark_validation": [
                ScientificBenchmarkResult(
                    benchmark_name="test",
                    domain=PhysicsDomain.QUANTUM_CHEMISTRY,
                    accuracy_score=0.6,
                    reference_value=1.0,
                    computed_value=1.2,
                    relative_error=0.2,
                    meets_accuracy_threshold=False,
                ),
            ],
        }

        recommendations = integrator.optimize_for_scientific_accuracy(
            sample_model_output, validation_results
        )

        assert isinstance(recommendations, dict)
        assert "optimization_type" in recommendations
        assert "recommendations" in recommendations
        assert "priority_actions" in recommendations
        assert recommendations["optimization_type"] == "scientific_accuracy"
        assert len(recommendations["recommendations"]) > 0
        assert len(recommendations["priority_actions"]) > 0


class TestScientificIntegrationWorkflow:
    """Integration tests for scientific computing components."""

    def test_complete_quantum_chemistry_workflow(self):
        """Test complete quantum chemistry validation workflow."""
        # Create quantum chemistry system
        integrator = ScientificComputingIntegrator(PhysicsDomain.QUANTUM_CHEMISTRY)

        # Simulate molecular orbital coefficients
        model_output = jnp.array(
            [
                [0.8, 0.6],  # MO 1
                [0.6, -0.8],  # MO 2
            ]
        )

        reference_data = {
            "physics_reference": {
                "energy": -76.4,  # Water energy in hartree
                "wavefunction": jnp.array([0.6, 0.8]),
                "symmetry_operations": [
                    {"type": "rotation", "axis": [0, 0, 1], "angle": jnp.pi},
                ],
            },
            "numerical_reference": model_output,
            "energy": jnp.array([-76.4]),
            "particle_number": jnp.array([10.0]),  # 10 electrons
        }

        benchmarks = {
            "water_total_energy": (-76.4, -76.41),  # Close to reference
            "dipole_moment": (1.85, 1.84),  # Debye
        }

        # Perform validation
        results = integrator.comprehensive_scientific_validation(
            model_output, reference_data, benchmarks
        )

        # Check results
        assert results["domain"] == "quantum_chemistry"
        assert "physics_metrics" in results
        assert "numerical_validation" in results
        assert "benchmark_validation" in results
        assert 0.0 <= results["overall_scientific_score"] <= 1.0

    def test_complete_fluid_dynamics_workflow(self):
        """Test complete fluid dynamics validation workflow."""
        # Create fluid dynamics system
        integrator = ScientificComputingIntegrator(PhysicsDomain.FLUID_DYNAMICS)

        # Simulate velocity field
        model_output = jnp.array(
            [
                [1.0, 0.0, 0.0],  # u, v, w at point 1
                [0.8, 0.2, 0.0],  # u, v, w at point 2
                [0.0, 0.0, 0.0],  # u, v, w at point 3 (boundary)
            ]
        )

        reference_data = {
            "physics_reference": {
                "momentum": [1.0, 0.1, 0.0],
                "mass": 1.0,
                "boundary_conditions": {
                    "dirichlet": {
                        "indices": [2],  # No-slip boundary
                        "values": [0.0],
                    },
                },
            },
            "numerical_reference": model_output,
            "momentum": jnp.array([1.0, 0.1, 0.0]),
            "mass": jnp.array([1.0]),
        }

        benchmarks = {
            "reynolds_stress": (0.1, 0.11),
            "pressure_drop": (1000.0, 995.0),
        }

        # Perform validation
        results = integrator.comprehensive_scientific_validation(
            model_output, reference_data, benchmarks
        )

        # Check results
        assert results["domain"] == "fluid_dynamics"
        assert "physics_metrics" in results
        assert 0.0 <= results["overall_scientific_score"] <= 1.0


class TestConservationLawImportFromCore:
    """Test that ConservationLaw is imported from core.physics, not duplicated."""

    def test_conservation_law_is_from_core_physics(self):
        """Verify ConservationLaw comes from core.physics module."""
        from opifex.core.physics import ConservationLaw as CoreConservationLaw

        # The ConservationLaw we imported should be the same as core.physics
        assert ConservationLaw is CoreConservationLaw
        assert ConservationLaw.__module__ == "opifex.core.physics.conservation"

    def test_all_conservation_laws_accessible(self):
        """Verify all conservation laws from core are accessible."""
        assert hasattr(ConservationLaw, "ENERGY")
        assert hasattr(ConservationLaw, "MOMENTUM")
        assert hasattr(ConservationLaw, "ANGULAR_MOMENTUM")
        assert hasattr(ConservationLaw, "MASS")
        assert hasattr(ConservationLaw, "CHARGE")
        assert hasattr(ConservationLaw, "PARTICLE_NUMBER")
        assert hasattr(ConservationLaw, "PROBABILITY")

    def test_physics_metrics_uses_core_conservation_law(self):
        """Test that PhysicsMetrics uses core.physics.ConservationLaw."""
        profiler = PhysicsProfiler(PhysicsDomain.QUANTUM_CHEMISTRY)

        model_output = jnp.array([1.0, 2.0, 3.0])
        reference_data = {
            "energy": 14.0,
            "momentum": [6.0, 0.0, 0.0],
            "total_mass": 6.0,
        }

        metrics = profiler.profile_physics_metrics(model_output, reference_data)

        # conservation_violations should use ConservationLaw enum as keys
        assert isinstance(metrics.conservation_violations, dict)
        for key in metrics.conservation_violations:
            assert isinstance(key, ConservationLaw)
            # Verify it's the core.physics enum
            assert key.__class__.__module__ == "opifex.core.physics.conservation"
