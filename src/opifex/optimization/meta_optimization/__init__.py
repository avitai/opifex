"""Meta-optimization algorithms for scientific machine learning.

This package implements meta-learning approaches to optimization including
learn-to-optimize (L2O) algorithms, adaptive learning rate scheduling,
warm-starting strategies, and performance monitoring. All implementations
follow FLAX NNX patterns and are designed for scientific computing applications.

Key Features:
    - Learn-to-optimize (L2O) meta-learning algorithms
    - Adaptive learning rate scheduling with multiple strategies
    - Warm-starting based on problem similarity
    - Performance monitoring and analytics
    - Quantum-aware optimization adaptations
    - Integration with existing training infrastructure

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from opifex.core.training.config import MetaOptimizerConfig

from .meta_optimizer import MetaOptimizer
from .monitoring import PerformanceMonitor
from .neural_learner import LearnToOptimize
from .schedulers import AdaptiveLearningRateScheduler
from .warm_starting import WarmStartingStrategy


__all__ = [
    "AdaptiveLearningRateScheduler",
    "LearnToOptimize",
    "MetaOptimizer",
    "MetaOptimizerConfig",
    "PerformanceMonitor",
    "WarmStartingStrategy",
]
