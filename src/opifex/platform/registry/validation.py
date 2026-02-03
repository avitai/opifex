"""Validation Engine for Neural Functional Quality Assurance.

Provides automated testing, performance benchmarking, and quality assurance
workflows for neural functionals in the Opifex registry.
"""

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp


class ValidationStatus(Enum):
    """Status of validation process."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class TestType(Enum):
    """Types of validation tests."""

    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    SECURITY = "security"
    DOCUMENTATION = "documentation"


@dataclass
class ValidationRule:
    """A validation rule for neural functionals."""

    name: str
    test_type: TestType
    description: str
    test_function: Callable
    required: bool = True
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation test."""

    rule_name: str
    status: ValidationStatus
    score: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_traceback: str | None = None


@dataclass
class FunctionalReport:
    """Complete validation report for a neural functional."""

    functional_id: str
    version: str
    overall_status: ValidationStatus
    overall_score: float
    test_results: list[ValidationResult] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    created_at: str = ""
    execution_time: float = 0.0


class ValidationEngine:
    """Neural functional validation engine.

    Provides comprehensive validation including functional testing,
    performance benchmarking, and quality assurance workflows.
    """

    def __init__(
        self,
        registry_service,
        enable_gpu_testing: bool = True,
        strict_mode: bool = False,
    ):
        """Initialize validation engine.

        Args:
            registry_service: Registry service for functional access
            enable_gpu_testing: Whether to run GPU performance tests
            strict_mode: Whether to fail on warnings
        """
        self.registry = registry_service
        self.enable_gpu = enable_gpu_testing
        self.strict_mode = strict_mode

        # Default validation rules
        self.rules: list[ValidationRule] = []
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        # Functional tests
        self.add_rule(
            ValidationRule(
                name="input_validation",
                test_type=TestType.FUNCTIONAL,
                description="Validate input parameter handling",
                test_function=self._test_input_validation,
                required=True,
            )
        )

        self.add_rule(
            ValidationRule(
                name="output_validation",
                test_type=TestType.FUNCTIONAL,
                description="Validate output format and consistency",
                test_function=self._test_output_validation,
                required=True,
            )
        )

        self.add_rule(
            ValidationRule(
                name="determinism_test",
                test_type=TestType.FUNCTIONAL,
                description="Test functional determinism with same inputs",
                test_function=self._test_determinism,
                required=False,
            )
        )

        # Performance tests
        self.add_rule(
            ValidationRule(
                name="memory_usage",
                test_type=TestType.PERFORMANCE,
                description="Measure memory consumption",
                test_function=self._test_memory_usage,
                required=True,
            )
        )

        self.add_rule(
            ValidationRule(
                name="execution_speed",
                test_type=TestType.PERFORMANCE,
                description="Measure execution performance",
                test_function=self._test_execution_speed,
                required=True,
            )
        )

        # Compatibility tests
        self.add_rule(
            ValidationRule(
                name="jax_compatibility",
                test_type=TestType.COMPATIBILITY,
                description="Test JAX/JIT compatibility",
                test_function=self._test_jax_compatibility,
                required=True,
            )
        )

        self.add_rule(
            ValidationRule(
                name="batch_processing",
                test_type=TestType.COMPATIBILITY,
                description="Test batch processing capabilities",
                test_function=self._test_batch_processing,
                required=False,
            )
        )

        # Documentation tests
        self.add_rule(
            ValidationRule(
                name="documentation_quality",
                test_type=TestType.DOCUMENTATION,
                description="Validate documentation completeness",
                test_function=self._test_documentation,
                required=False,
            )
        )

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was removed
        """
        initial_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        return len(self.rules) < initial_count

    async def validate_functional(
        self,
        functional_id: str,
        version: str,
        test_types: list[TestType] | None = None,
        include_performance: bool = True,
    ) -> FunctionalReport:
        """Validate a neural functional comprehensively.

        Args:
            functional_id: ID of functional to validate
            version: Version to validate
            test_types: Types of tests to run (all if None)
            include_performance: Whether to include performance tests

        Returns:
            Complete validation report
        """
        start_time = time.time()

        # Load functional
        functional_data = await self.registry.retrieve_functional(
            functional_id, version
        )
        if not functional_data:
            return FunctionalReport(
                functional_id=functional_id,
                version=version,
                overall_status=ValidationStatus.FAILED,
                overall_score=0.0,
                test_results=[
                    ValidationResult(
                        rule_name="functional_loading",
                        status=ValidationStatus.FAILED,
                        message="Could not load functional",
                    )
                ],
            )

        # Filter rules by test types
        rules_to_run = self.rules
        if test_types:
            rules_to_run = [rule for rule in self.rules if rule.test_type in test_types]

        if not include_performance:
            rules_to_run = [
                rule for rule in rules_to_run if rule.test_type != TestType.PERFORMANCE
            ]

        # Run validation tests
        test_results = []
        for rule in rules_to_run:
            result = await self._run_validation_rule(rule, functional_data)
            test_results.append(result)

        # Calculate overall score and status
        overall_score = self._calculate_overall_score(test_results)
        overall_status = self._determine_overall_status(test_results)

        # Generate performance metrics
        performance_metrics = self._extract_performance_metrics(test_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, functional_data)

        execution_time = time.time() - start_time

        return FunctionalReport(
            functional_id=functional_id,
            version=version,
            overall_status=overall_status,
            overall_score=overall_score,
            test_results=test_results,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            execution_time=execution_time,
        )

    async def _run_validation_rule(
        self, rule: ValidationRule, functional_data: dict[str, Any]
    ) -> ValidationResult:
        """Run a single validation rule.

        Args:
            rule: Validation rule to execute
            functional_data: Functional data to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        try:
            # Execute test function with timeout handling
            result = await self._execute_with_timeout(
                rule.test_function, functional_data, rule.timeout_seconds
            )

            execution_time = time.time() - start_time

            return ValidationResult(
                rule_name=rule.name,
                status=result.get("status", ValidationStatus.PASSED),
                score=result.get("score", 1.0),
                message=result.get("message", "Test passed"),
                details=result.get("details", {}),
                execution_time=execution_time,
            )

        except TimeoutError:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                score=0.0,
                message=f"Test timed out after {rule.timeout_seconds} seconds",
                execution_time=rule.timeout_seconds,
            )

        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                score=0.0,
                message=f"Test failed: {e!s}",
                error_traceback=traceback.format_exc(),
                execution_time=time.time() - start_time,
            )

    async def _execute_with_timeout(
        self, test_function: Callable, functional_data: dict[str, Any], timeout: int
    ) -> dict[str, Any]:
        """Execute test function with timeout.

        Args:
            test_function: Function to execute
            functional_data: Functional data
            timeout: Timeout in seconds

        Returns:
            Test result dictionary
        """
        # Simple implementation - in production would use asyncio.wait_for
        result = test_function(functional_data)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def _calculate_overall_score(self, results: list[ValidationResult]) -> float:
        """Calculate overall validation score.

        Args:
            results: List of validation results

        Returns:
            Overall score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Weight scores by requirement status
        total_score = 0.0
        total_weight = 0.0

        for result in results:
            # Find corresponding rule to check if required
            rule = next((r for r in self.rules if r.name == result.rule_name), None)
            weight = 2.0 if rule and rule.required else 1.0

            total_score += result.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_overall_status(
        self, results: list[ValidationResult]
    ) -> ValidationStatus:
        """Determine overall validation status.

        Args:
            results: List of validation results

        Returns:
            Overall validation status
        """
        if not results:
            return ValidationStatus.FAILED

        # Check for any failed required tests
        for result in results:
            rule = next((r for r in self.rules if r.name == result.rule_name), None)
            if rule and rule.required and result.status == ValidationStatus.FAILED:
                return ValidationStatus.FAILED

        # Check if all passed
        if all(r.status == ValidationStatus.PASSED for r in results):
            return ValidationStatus.PASSED

        # Check for warnings
        if any(r.status == ValidationStatus.WARNING for r in results):
            return (
                ValidationStatus.WARNING
                if not self.strict_mode
                else ValidationStatus.FAILED
            )

        return ValidationStatus.PASSED

    def _extract_performance_metrics(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """Extract performance metrics from test results.

        Args:
            results: List of validation results

        Returns:
            Performance metrics dictionary
        """
        metrics = {}

        for result in results:
            if result.details:
                # Extract execution time
                if "execution_time" in result.details:
                    metrics[f"{result.rule_name}_execution_time"] = result.details[
                        "execution_time"
                    ]

                # Extract memory usage
                if "memory_mb" in result.details:
                    metrics[f"{result.rule_name}_memory_mb"] = result.details[
                        "memory_mb"
                    ]

                # Extract throughput
                if "throughput" in result.details:
                    metrics[f"{result.rule_name}_throughput"] = result.details[
                        "throughput"
                    ]

        return metrics

    def _generate_recommendations(
        self, results: list[ValidationResult], functional_data: dict[str, Any]
    ) -> list[str]:
        """Generate improvement recommendations.

        Args:
            results: Validation results
            functional_data: Functional data

        Returns:
            List of recommendations
        """
        recommendations = []

        for result in results:
            if result.status == ValidationStatus.FAILED:
                recommendations.append(f"Fix {result.rule_name}: {result.message}")
            elif result.status == ValidationStatus.WARNING:
                recommendations.append(
                    f"Consider improving {result.rule_name}: {result.message}"
                )

        # Performance recommendations
        memory_results = [
            r for r in results if "memory" in r.rule_name and r.details.get("memory_mb")
        ]
        if memory_results:
            max_memory = max(r.details["memory_mb"] for r in memory_results)
            if max_memory > 1000:  # > 1GB
                recommendations.append(
                    "Consider memory optimization - high memory usage detected"
                )

        execution_results = [
            r
            for r in results
            if "speed" in r.rule_name and r.details.get("execution_time")
        ]
        if execution_results:
            max_time = max(r.details["execution_time"] for r in execution_results)
            if max_time > 10.0:  # > 10 seconds
                recommendations.append(
                    "Consider performance optimization - slow execution detected"
                )

        return recommendations

    # Validation test implementations
    def _test_input_validation(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test input parameter validation."""
        try:
            # Test with various input types and sizes
            test_inputs = [
                jnp.array([1.0, 2.0, 3.0]),
                jnp.ones((10, 10)),
                jnp.zeros((5,)),
            ]

            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            for test_input in test_inputs:
                try:
                    _ = functional(test_input)
                except Exception as e:
                    return {
                        "status": ValidationStatus.WARNING,
                        "message": f"Input validation concern: {e!s}",
                        "score": 0.7,
                    }

            return {
                "status": ValidationStatus.PASSED,
                "message": "Input validation passed",
                "score": 1.0,
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Input validation failed: {e!s}",
                "score": 0.0,
            }

    def _test_output_validation(
        self, functional_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Test output format validation."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Test with standard input
            test_input = jnp.ones((5, 5))
            output = functional(test_input)

            # Check output is JAX array
            if not isinstance(output, jnp.ndarray):
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "Output is not a JAX array",
                    "score": 0.0,
                }

            # Check for NaN or infinite values
            if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
                return {
                    "status": ValidationStatus.WARNING,
                    "message": "Output contains NaN or infinite values",
                    "score": 0.5,
                }

            return {
                "status": ValidationStatus.PASSED,
                "message": "Output validation passed",
                "score": 1.0,
                "details": {
                    "output_shape": output.shape,
                    "output_dtype": str(output.dtype),
                },
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Output validation failed: {e!s}",
                "score": 0.0,
            }

    def _test_determinism(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test functional determinism."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Run same input multiple times
            test_input = jnp.ones((3, 3))
            outputs = []

            for _ in range(3):
                output = functional(test_input)
                outputs.append(output)

            # Check if outputs are identical
            for i in range(1, len(outputs)):
                if not jnp.allclose(outputs[0], outputs[i], rtol=1e-10):
                    return {
                        "status": ValidationStatus.WARNING,
                        "message": "Functional may not be deterministic",
                        "score": 0.8,
                    }

            return {
                "status": ValidationStatus.PASSED,
                "message": "Determinism test passed",
                "score": 1.0,
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Determinism test failed: {e!s}",
                "score": 0.0,
            }

    def _test_memory_usage(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test memory usage."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Simple memory estimation (simplified implementation)
            test_input = jnp.ones((100, 100))

            # Estimate memory usage
            input_memory = test_input.nbytes / (1024 * 1024)  # MB

            output = functional(test_input)
            output_memory = output.nbytes / (1024 * 1024)  # MB

            total_memory = input_memory + output_memory

            status = ValidationStatus.PASSED
            message = "Memory usage within limits"

            if total_memory > 100:  # > 100MB
                status = ValidationStatus.WARNING
                message = "High memory usage detected"

            return {
                "status": status,
                "message": message,
                "score": max(0.1, 1.0 - (total_memory / 1000)),  # Score based on memory
                "details": {"memory_mb": total_memory},
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Memory test failed: {e!s}",
                "score": 0.0,
            }

    def _test_execution_speed(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test execution speed."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Performance test with timing
            test_input = jnp.ones((50, 50))

            # Warm-up run
            _ = functional(test_input)

            # Timed runs
            times = []
            for _ in range(5):
                start_time = time.time()
                _ = functional(test_input)
                execution_time = time.time() - start_time
                times.append(execution_time)

            avg_time = sum(times) / len(times)

            status = ValidationStatus.PASSED
            message = "Execution speed acceptable"
            score = 1.0

            if avg_time > 1.0:  # > 1 second
                status = ValidationStatus.WARNING
                message = "Slow execution detected"
                score = max(0.1, 1.0 - avg_time / 10.0)

            return {
                "status": status,
                "message": message,
                "score": score,
                "details": {
                    "execution_time": avg_time,
                    "min_time": min(times),
                    "max_time": max(times),
                },
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Speed test failed: {e!s}",
                "score": 0.0,
            }

    def _test_jax_compatibility(
        self, functional_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Test JAX/JIT compatibility."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Test JIT compilation
            test_input = jnp.ones((5, 5))

            try:
                jit_functional = jax.jit(functional)
                _ = jit_functional(test_input)

                return {
                    "status": ValidationStatus.PASSED,
                    "message": "JAX/JIT compatibility confirmed",
                    "score": 1.0,
                    "details": {"jit_compatible": True},
                }

            except Exception as jit_error:
                return {
                    "status": ValidationStatus.WARNING,
                    "message": f"JIT compilation failed: {jit_error!s}",
                    "score": 0.7,
                    "details": {"jit_compatible": False},
                }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"JAX compatibility test failed: {e!s}",
                "score": 0.0,
            }

    def _test_batch_processing(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test batch processing capabilities."""
        try:
            functional = functional_data.get("functional")
            if not functional:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": "No functional found",
                }

            # Test with batched input
            single_input = jnp.ones((5, 5))
            batch_input = jnp.ones((3, 5, 5))  # Batch of 3

            try:
                # Test single input
                single_output = functional(single_input)

                # Test batch input
                batch_output = functional(batch_input)

                # Check if batch dimension is preserved
                expected_batch_shape = (3, *single_output.shape)
                if batch_output.shape != expected_batch_shape:
                    return {
                        "status": ValidationStatus.WARNING,
                        "message": "Batch processing may not work as expected",
                        "score": 0.6,
                    }

                return {
                    "status": ValidationStatus.PASSED,
                    "message": "Batch processing supported",
                    "score": 1.0,
                    "details": {"batch_compatible": True},
                }

            except Exception as batch_error:
                return {
                    "status": ValidationStatus.WARNING,
                    "message": f"Batch processing not supported: {batch_error!s}",
                    "score": 0.8,
                    "details": {"batch_compatible": False},
                }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Batch processing test failed: {e!s}",
                "score": 0.0,
            }

    def _test_documentation(self, functional_data: dict[str, Any]) -> dict[str, Any]:
        """Test documentation quality."""
        try:
            # Check for documentation fields
            required_fields = ["name", "description", "type"]
            optional_fields = ["author", "examples", "parameters"]

            missing_required = []
            missing_optional = []

            for field in required_fields:
                if not functional_data.get(field):
                    missing_required.append(field)

            for field in optional_fields:
                if not functional_data.get(field):
                    missing_optional.append(field)

            if missing_required:
                return {
                    "status": ValidationStatus.FAILED,
                    "message": (
                        f"Missing required documentation: {', '.join(missing_required)}"
                    ),
                    "score": 0.0,
                }

            # Calculate score based on completeness
            total_fields = len(required_fields) + len(optional_fields)
            missing_count = len(missing_optional)
            score = (total_fields - missing_count) / total_fields

            status = ValidationStatus.PASSED
            message = "Documentation complete"

            if missing_optional:
                status = ValidationStatus.WARNING
                message = (
                    f"Missing optional documentation: {', '.join(missing_optional)}"
                )

            return {
                "status": status,
                "message": message,
                "score": score,
                "details": {
                    "missing_required": missing_required,
                    "missing_optional": missing_optional,
                },
            }

        except Exception as e:
            return {
                "status": ValidationStatus.FAILED,
                "message": f"Documentation test failed: {e!s}",
                "score": 0.0,
            }
