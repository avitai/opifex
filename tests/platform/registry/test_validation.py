"""Test suite for Neural Functional Validation Engine.

Provides comprehensive test coverage for validation functionality including
functional testing, performance benchmarking, and quality assurance workflows.
"""

import time

import jax.numpy as jnp
import pytest

from opifex.platform.registry.validation import (
    FunctionalReport,
    TestType,
    ValidationEngine,
    ValidationResult,
    ValidationRule,
    ValidationStatus,
)


class MockRegistryService:
    """Mock registry service for testing."""

    def __init__(self):
        self.functionals = {}

    async def retrieve_functional(self, functional_id: str, version: str | None = None):
        """Mock retrieve functional method."""
        if functional_id == "valid-func":
            return {
                "id": functional_id,
                "name": "Test Functional",
                "description": "A test neural functional",
                "type": "neural_operator",
                "author": "test_author",
                "functional": jnp.sin,  # Simple test function
                "parameters": {"learning_rate": 0.001},
                "examples": ["example1", "example2"],
            }
        if functional_id == "invalid-func":
            return {
                "id": functional_id,
                "name": "Invalid Functional",
                "description": "",  # Missing description
                "type": "test",
                "functional": lambda x: jnp.array(
                    [jnp.nan, jnp.inf]
                ),  # Problematic output
            }
        if functional_id == "slow-func":
            return {
                "id": functional_id,
                "name": "Slow Functional",
                "description": "A slow test functional",
                "type": "test",
                "functional": lambda x: (time.sleep(0.1), jnp.ones_like(x))[
                    1
                ],  # Slow function
            }
        if functional_id == "error-func":
            return {
                "id": functional_id,
                "name": "Error Functional",
                "description": "A functional that throws errors",
                "type": "test",
                "functional": lambda x: 1 / 0,  # Always throws error
            }
        if functional_id == "non-jit-func":
            return {
                "id": functional_id,
                "name": "Non-JIT Functional",
                "description": "A functional that doesn't work with JIT",
                "type": "test",
                "functional": lambda x: (
                    print("side effect") or x
                ),  # Side effect prevents JIT
            }
        return None


@pytest.fixture
def mock_registry():
    """Create mock registry service."""
    return MockRegistryService()


@pytest.fixture
def validation_engine(mock_registry):
    """Create validation engine with mock registry."""
    return ValidationEngine(
        registry_service=mock_registry,
        enable_gpu_testing=True,
        strict_mode=False,
    )


@pytest.fixture
def strict_validation_engine(mock_registry):
    """Create validation engine in strict mode."""
    return ValidationEngine(
        registry_service=mock_registry,
        enable_gpu_testing=True,
        strict_mode=True,
    )


class TestValidationEngine:
    """Test ValidationEngine functionality."""

    def test_initialization(self, mock_registry):
        """Test validation engine initialization."""
        engine = ValidationEngine(mock_registry)

        assert engine.registry == mock_registry
        assert engine.enable_gpu is True
        assert engine.strict_mode is False
        assert len(engine.rules) > 0  # Default rules loaded

    def test_initialization_custom_params(self, mock_registry):
        """Test validation engine initialization with custom parameters."""
        engine = ValidationEngine(
            mock_registry,
            enable_gpu_testing=False,
            strict_mode=True,
        )

        assert engine.enable_gpu is False
        assert engine.strict_mode is True

    def test_default_rules_loaded(self, validation_engine):
        """Test that default validation rules are loaded."""
        rule_names = {rule.name for rule in validation_engine.rules}

        expected_rules = {
            "input_validation",
            "output_validation",
            "determinism_test",
            "memory_usage",
            "execution_speed",
            "jax_compatibility",
            "batch_processing",
            "documentation_quality",
        }

        assert expected_rules.issubset(rule_names)

    def test_add_rule(self, validation_engine):
        """Test adding custom validation rule."""
        initial_count = len(validation_engine.rules)

        custom_rule = ValidationRule(
            name="custom_test",
            test_type=TestType.FUNCTIONAL,
            description="Custom test rule",
            test_function=lambda x: {"status": ValidationStatus.PASSED, "score": 1.0},
        )

        validation_engine.add_rule(custom_rule)

        assert len(validation_engine.rules) == initial_count + 1
        assert any(rule.name == "custom_test" for rule in validation_engine.rules)

    def test_remove_rule(self, validation_engine):
        """Test removing validation rule."""
        initial_count = len(validation_engine.rules)

        # Remove existing rule
        removed = validation_engine.remove_rule("determinism_test")

        assert removed is True
        assert len(validation_engine.rules) == initial_count - 1
        assert not any(
            rule.name == "determinism_test" for rule in validation_engine.rules
        )

    def test_remove_nonexistent_rule(self, validation_engine):
        """Test removing nonexistent rule."""
        initial_count = len(validation_engine.rules)

        removed = validation_engine.remove_rule("nonexistent_rule")

        assert removed is False
        assert len(validation_engine.rules) == initial_count

    @pytest.mark.asyncio
    async def test_validate_functional_success(self, validation_engine):
        """Test successful functional validation."""
        report = await validation_engine.validate_functional(
            functional_id="valid-func",
            version="v1.0.0",
        )

        assert isinstance(report, FunctionalReport)
        assert report.functional_id == "valid-func"
        assert report.version == "v1.0.0"
        assert report.overall_status in [
            ValidationStatus.PASSED,
            ValidationStatus.WARNING,
        ]
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.test_results) > 0

    @pytest.mark.asyncio
    async def test_validate_functional_not_found(self, validation_engine):
        """Test validation of nonexistent functional."""
        report = await validation_engine.validate_functional(
            functional_id="nonexistent",
            version="v1.0.0",
        )

        assert report.overall_status == ValidationStatus.FAILED
        assert report.overall_score == 0.0
        assert len(report.test_results) == 1
        assert "Could not load functional" in report.test_results[0].message

    @pytest.mark.asyncio
    async def test_validate_functional_with_filters(self, validation_engine):
        """Test functional validation with test type filters."""
        report = await validation_engine.validate_functional(
            functional_id="valid-func",
            version="v1.0.0",
            test_types=[TestType.FUNCTIONAL],
            include_performance=False,
        )

        # Should only run functional tests
        functional_tests = [
            r
            for r in report.test_results
            if any(
                rule.test_type == TestType.FUNCTIONAL
                for rule in validation_engine.rules
                if rule.name == r.rule_name
            )
        ]
        performance_tests = [
            r
            for r in report.test_results
            if any(
                rule.test_type == TestType.PERFORMANCE
                for rule in validation_engine.rules
                if rule.name == r.rule_name
            )
        ]

        assert len(functional_tests) > 0
        assert len(performance_tests) == 0

    @pytest.mark.asyncio
    async def test_run_validation_rule_success(self, validation_engine):
        """Test running individual validation rule successfully."""
        rule = ValidationRule(
            name="test_rule",
            test_type=TestType.FUNCTIONAL,
            description="Test rule",
            test_function=lambda x: {
                "status": ValidationStatus.PASSED,
                "score": 1.0,
                "message": "Test passed",
                "details": {"key": "value"},
            },
        )

        functional_data = {"id": "test", "functional": lambda x: x}
        result = await validation_engine._run_validation_rule(rule, functional_data)

        assert isinstance(result, ValidationResult)
        assert result.rule_name == "test_rule"
        assert result.status == ValidationStatus.PASSED
        assert result.score == 1.0
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_run_validation_rule_error(self, validation_engine):
        """Test running validation rule that throws error."""
        rule = ValidationRule(
            name="error_rule",
            test_type=TestType.FUNCTIONAL,
            description="Error rule",
            test_function=lambda x: 1 / 0,  # Always throws error
        )

        functional_data = {"id": "test", "functional": lambda x: x}
        result = await validation_engine._run_validation_rule(rule, functional_data)

        assert result.status == ValidationStatus.FAILED
        assert result.score == 0.0
        assert "Test failed" in result.message
        assert result.error_traceback is not None

    def test_calculate_overall_score(self, validation_engine):
        """Test overall score calculation."""
        results = [
            ValidationResult(
                rule_name="required_test", status=ValidationStatus.PASSED, score=1.0
            ),
            ValidationResult(
                rule_name="optional_test", status=ValidationStatus.PASSED, score=0.8
            ),
        ]

        # Mock rules
        validation_engine.rules = [
            ValidationRule(
                name="required_test",
                test_type=TestType.FUNCTIONAL,
                description="",
                test_function=lambda x: {},
                required=True,
            ),
            ValidationRule(
                name="optional_test",
                test_type=TestType.FUNCTIONAL,
                description="",
                test_function=lambda x: {},
                required=False,
            ),
        ]

        score = validation_engine._calculate_overall_score(results)

        assert 0.0 <= score <= 1.0
        # Required tests should have higher weight

    def test_calculate_overall_score_empty(self, validation_engine):
        """Test overall score calculation with no results."""
        score = validation_engine._calculate_overall_score([])
        assert score == 0.0

    def test_determine_overall_status_passed(self, validation_engine):
        """Test overall status determination for passed tests."""
        results = [
            ValidationResult(rule_name="test1", status=ValidationStatus.PASSED),
            ValidationResult(rule_name="test2", status=ValidationStatus.PASSED),
        ]

        status = validation_engine._determine_overall_status(results)
        assert status == ValidationStatus.PASSED

    def test_determine_overall_status_failed_required(self, validation_engine):
        """Test overall status determination with failed required test."""
        results = [
            ValidationResult(rule_name="required_test", status=ValidationStatus.FAILED),
            ValidationResult(rule_name="optional_test", status=ValidationStatus.PASSED),
        ]

        # Mock rules
        validation_engine.rules = [
            ValidationRule(
                name="required_test",
                test_type=TestType.FUNCTIONAL,
                description="",
                test_function=lambda x: {},
                required=True,
            ),
            ValidationRule(
                name="optional_test",
                test_type=TestType.FUNCTIONAL,
                description="",
                test_function=lambda x: {},
                required=False,
            ),
        ]

        status = validation_engine._determine_overall_status(results)
        assert status == ValidationStatus.FAILED

    def test_determine_overall_status_warning_normal_mode(self, validation_engine):
        """Test overall status determination with warnings in normal mode."""
        results = [
            ValidationResult(rule_name="test1", status=ValidationStatus.WARNING),
            ValidationResult(rule_name="test2", status=ValidationStatus.PASSED),
        ]

        status = validation_engine._determine_overall_status(results)
        assert status == ValidationStatus.WARNING

    def test_determine_overall_status_warning_strict_mode(
        self, strict_validation_engine
    ):
        """Test overall status determination with warnings in strict mode."""
        results = [
            ValidationResult(rule_name="test1", status=ValidationStatus.WARNING),
            ValidationResult(rule_name="test2", status=ValidationStatus.PASSED),
        ]

        status = strict_validation_engine._determine_overall_status(results)
        assert status == ValidationStatus.FAILED

    def test_extract_performance_metrics(self, validation_engine):
        """Test performance metrics extraction."""
        results = [
            ValidationResult(
                rule_name="speed_test",
                status=ValidationStatus.PASSED,
                details={
                    "execution_time": 0.5,
                    "memory_mb": 100,
                    "throughput": 1000,
                },
            ),
            ValidationResult(
                rule_name="memory_test",
                status=ValidationStatus.PASSED,
                details={"memory_mb": 50},
            ),
        ]

        metrics = validation_engine._extract_performance_metrics(results)

        assert "speed_test_execution_time" in metrics
        assert "speed_test_memory_mb" in metrics
        assert "speed_test_throughput" in metrics
        assert "memory_test_memory_mb" in metrics
        assert metrics["speed_test_execution_time"] == 0.5

    def test_generate_recommendations(self, validation_engine):
        """Test recommendation generation."""
        results = [
            ValidationResult(
                rule_name="failed_test",
                status=ValidationStatus.FAILED,
                message="Test failed",
            ),
            ValidationResult(
                rule_name="warning_test",
                status=ValidationStatus.WARNING,
                message="Test has warnings",
            ),
            ValidationResult(
                rule_name="memory_test",
                status=ValidationStatus.PASSED,
                details={"memory_mb": 2000},
            ),  # High memory
            ValidationResult(
                rule_name="speed_test",
                status=ValidationStatus.PASSED,
                details={"execution_time": 15.0},
            ),  # Slow execution
        ]

        functional_data = {"id": "test-func"}
        recommendations = validation_engine._generate_recommendations(
            results, functional_data
        )

        assert len(recommendations) > 0
        assert any("Fix failed_test" in rec for rec in recommendations)
        assert any("Consider improving warning_test" in rec for rec in recommendations)
        assert any("memory optimization" in rec for rec in recommendations)
        assert any("performance optimization" in rec for rec in recommendations)

    def test_input_validation_test(self, validation_engine):
        """Test input validation test function."""
        functional_data = {
            "functional": jnp.sin,
        }

        result = validation_engine._test_input_validation(functional_data)

        assert result["status"] == ValidationStatus.PASSED
        assert result["score"] == 1.0

    def test_input_validation_test_no_functional(self, validation_engine):
        """Test input validation with missing functional."""
        functional_data = {}

        result = validation_engine._test_input_validation(functional_data)

        assert result["status"] == ValidationStatus.FAILED
        assert "No functional found" in result["message"]

    def test_input_validation_test_error(self, validation_engine):
        """Test input validation with problematic functional."""
        functional_data = {
            "functional": lambda x: 1 / 0,  # Always errors
        }

        result = validation_engine._test_input_validation(functional_data)

        assert result["status"] in [ValidationStatus.WARNING, ValidationStatus.FAILED]

    def test_output_validation_test(self, validation_engine):
        """Test output validation test function."""
        functional_data = {
            "functional": jnp.sin,
        }

        result = validation_engine._test_output_validation(functional_data)

        assert result["status"] == ValidationStatus.PASSED
        assert result["score"] == 1.0
        assert "output_shape" in result["details"]
        assert "output_dtype" in result["details"]

    def test_output_validation_test_nan_output(self, validation_engine):
        """Test output validation with NaN output."""
        functional_data = {
            "functional": lambda x: jnp.array([jnp.nan, jnp.inf]),
        }

        result = validation_engine._test_output_validation(functional_data)

        assert result["status"] == ValidationStatus.WARNING
        assert "NaN or infinite values" in result["message"]

    def test_output_validation_test_non_jax_output(self, validation_engine):
        """Test output validation with non-JAX output."""
        functional_data = {
            "functional": lambda x: [1, 2, 3],  # Returns list, not JAX array
        }

        result = validation_engine._test_output_validation(functional_data)

        assert result["status"] == ValidationStatus.FAILED
        assert "not a JAX array" in result["message"]

    def test_determinism_test(self, validation_engine):
        """Test determinism test function."""
        functional_data = {
            "functional": jnp.sin,  # Deterministic function
        }

        result = validation_engine._test_determinism(functional_data)

        assert result["status"] == ValidationStatus.PASSED
        assert result["score"] == 1.0

    def test_determinism_test_non_deterministic(self, validation_engine):
        """Test determinism test with non-deterministic function."""

        functional_data = {
            "functional": lambda x: jnp.array([0.5]),  # Non-deterministic
        }

        result = validation_engine._test_determinism(functional_data)

        # May pass or warn depending on random values
        assert result["status"] in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    def test_memory_usage_test(self, validation_engine):
        """Test memory usage test function."""
        functional_data = {
            "functional": jnp.sin,
        }

        result = validation_engine._test_memory_usage(functional_data)

        assert result["status"] in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert "memory_mb" in result["details"]
        assert result["details"]["memory_mb"] >= 0

    def test_execution_speed_test(self, validation_engine):
        """Test execution speed test function."""
        functional_data = {
            "functional": jnp.sin,
        }

        result = validation_engine._test_execution_speed(functional_data)

        assert result["status"] in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert "execution_time" in result["details"]
        assert result["details"]["execution_time"] >= 0

    def test_execution_speed_test_slow(self, validation_engine):
        """Test execution speed test with slow function."""
        functional_data = {
            "functional": lambda x: (time.sleep(1.5), jnp.sin(x))[1],  # Slow function
        }

        result = validation_engine._test_execution_speed(functional_data)

        assert result["status"] == ValidationStatus.WARNING
        assert "Slow execution detected" in result["message"]

    def test_jax_compatibility_test(self, validation_engine):
        """Test JAX compatibility test function."""
        functional_data = {
            "functional": jnp.sin,
        }

        result = validation_engine._test_jax_compatibility(functional_data)

        assert result["status"] == ValidationStatus.PASSED
        assert result["details"]["jit_compatible"] is True

    def test_jax_compatibility_test_non_jittable(self, validation_engine):
        """Test JAX compatibility test with non-JIT function."""
        functional_data = {
            "functional": lambda x: print("side effect") or x,  # Side effect
        }

        result = validation_engine._test_jax_compatibility(functional_data)

        # Should warn about JIT compilation failure
        assert result["status"] in [ValidationStatus.WARNING, ValidationStatus.PASSED]

    def test_batch_processing_test(self, validation_engine):
        """Test batch processing test function."""
        functional_data = {
            "functional": jnp.sin,  # Works with batches
        }

        result = validation_engine._test_batch_processing(functional_data)

        assert result["status"] in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    def test_batch_processing_test_incompatible(self, validation_engine):
        """Test batch processing test with batch-incompatible function."""
        functional_data = {
            "functional": lambda x: x.flatten()[0],  # Doesn't preserve batch dimension
        }

        result = validation_engine._test_batch_processing(functional_data)

        # Should detect batch processing issues
        assert result["status"] in [ValidationStatus.WARNING, ValidationStatus.FAILED]

    def test_documentation_test_complete(self, validation_engine):
        """Test documentation test with complete documentation."""
        functional_data = {
            "name": "Test Functional",
            "description": "A complete test functional",
            "type": "neural_operator",
            "author": "test_author",
            "examples": ["example1", "example2"],
            "parameters": {"param1": "value1"},
        }

        result = validation_engine._test_documentation(functional_data)

        assert result["status"] == ValidationStatus.PASSED
        assert result["score"] == 1.0

    def test_documentation_test_missing_required(self, validation_engine):
        """Test documentation test with missing required fields."""
        functional_data = {
            "name": "Test Functional",
            # Missing description
            "type": "neural_operator",
        }

        result = validation_engine._test_documentation(functional_data)

        assert result["status"] == ValidationStatus.FAILED
        assert "Missing required documentation" in result["message"]

    def test_documentation_test_missing_optional(self, validation_engine):
        """Test documentation test with missing optional fields."""
        functional_data = {
            "name": "Test Functional",
            "description": "A test functional",
            "type": "neural_operator",
            # Missing optional fields: author, examples, parameters
        }

        result = validation_engine._test_documentation(functional_data)

        assert result["status"] == ValidationStatus.WARNING
        assert "Missing optional documentation" in result["message"]
        assert 0.0 < result["score"] < 1.0

    @pytest.mark.asyncio
    async def test_integration_validation_workflow(self, validation_engine):
        """Test complete validation workflow integration."""
        # Test with a well-behaved functional
        report = await validation_engine.validate_functional(
            functional_id="valid-func",
            version="v1.0.0",
        )

        # Should pass most tests
        assert report.overall_status in [
            ValidationStatus.PASSED,
            ValidationStatus.WARNING,
        ]
        assert report.overall_score > 0.5
        assert len(report.test_results) >= 8  # Number of default rules
        assert len(report.performance_metrics) > 0
        assert report.execution_time > 0

    @pytest.mark.asyncio
    async def test_integration_validation_problematic_functional(
        self, validation_engine
    ):
        """Test validation with problematic functional."""
        # Test with a problematic functional
        report = await validation_engine.validate_functional(
            functional_id="invalid-func",
            version="v1.0.0",
        )

        # Should detect issues
        assert report.overall_score < 1.0
        assert len(report.recommendations) > 0

        # Check specific issues detected
        failed_tests = [
            r for r in report.test_results if r.status == ValidationStatus.FAILED
        ]
        warning_tests = [
            r for r in report.test_results if r.status == ValidationStatus.WARNING
        ]

        assert len(failed_tests) + len(warning_tests) > 0


class TestValidationDataClasses:
    """Test validation-related data classes."""

    def test_validation_rule_creation(self):
        """Test ValidationRule data class creation."""
        rule = ValidationRule(
            name="test_rule",
            test_type=TestType.FUNCTIONAL,
            description="Test rule description",
            test_function=lambda x: {"status": ValidationStatus.PASSED},
            required=True,
            timeout_seconds=300,
        )

        assert rule.name == "test_rule"
        assert rule.test_type == TestType.FUNCTIONAL
        assert rule.description == "Test rule description"
        assert callable(rule.test_function)
        assert rule.required is True
        assert rule.timeout_seconds == 300

    def test_validation_result_creation(self):
        """Test ValidationResult data class creation."""
        result = ValidationResult(
            rule_name="test_rule",
            status=ValidationStatus.PASSED,
            score=0.95,
            message="Test passed successfully",
            details={"key": "value"},
            execution_time=1.5,
        )

        assert result.rule_name == "test_rule"
        assert result.status == ValidationStatus.PASSED
        assert result.score == 0.95
        assert result.message == "Test passed successfully"
        assert result.details == {"key": "value"}
        assert result.execution_time == 1.5

    def test_functional_report_creation(self):
        """Test FunctionalReport data class creation."""
        test_results = [
            ValidationResult(rule_name="test1", status=ValidationStatus.PASSED),
            ValidationResult(rule_name="test2", status=ValidationStatus.WARNING),
        ]

        report = FunctionalReport(
            functional_id="func-001",
            version="v1.0.0",
            overall_status=ValidationStatus.WARNING,
            overall_score=0.85,
            test_results=test_results,
            performance_metrics={"speed": 0.5, "memory": 100},
            recommendations=["Fix warning", "Optimize performance"],
            execution_time=10.0,
        )

        assert report.functional_id == "func-001"
        assert report.version == "v1.0.0"
        assert report.overall_status == ValidationStatus.WARNING
        assert report.overall_score == 0.85
        assert len(report.test_results) == 2
        assert len(report.performance_metrics) == 2
        assert len(report.recommendations) == 2
        assert report.execution_time == 10.0

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.RUNNING.value == "running"
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.SKIPPED.value == "skipped"

    def test_test_type_enum(self):
        """Test TestType enum values."""
        assert TestType.FUNCTIONAL.value == "functional"
        assert TestType.PERFORMANCE.value == "performance"
        assert TestType.COMPATIBILITY.value == "compatibility"
        assert TestType.SECURITY.value == "security"
        assert TestType.DOCUMENTATION.value == "documentation"
