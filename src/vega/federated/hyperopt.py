"""
Federated Hyperparameter Optimization

Advanced distributed Bayesian optimization for hyperparameter tuning across federated learning participants.
Implements Gaussian Process surrogate models with federated knowledge sharing, acquisition function optimization,
and distributed search coordination.

Features:
- Distributed Bayesian optimization with participant coordination
- Multiple acquisition functions (EI, UCB, PI)
- Gaussian Process surrogate models with federated knowledge
- Hyperparameter space definition and sampling
- Multi-objective optimization support
- Privacy-preserving optimization with differential privacy
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


logger = logging.getLogger(__name__)


class HyperparameterType(Enum):
    """Types of hyperparameters for optimization."""

    FLOAT = "float"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


class AcquisitionFunction(Enum):
    """Acquisition functions for Bayesian optimization."""

    EXPECTED_IMPROVEMENT = "ei"
    UPPER_CONFIDENCE_BOUND = "ucb"
    PROBABILITY_OF_IMPROVEMENT = "pi"
    ENTROPY_SEARCH = "es"


@dataclass
class HyperparameterDimension:
    """Defines a single hyperparameter dimension."""

    name: str
    param_type: HyperparameterType
    bounds: Tuple[float, float] = None  # For continuous parameters
    choices: List[Any] = None  # For categorical parameters
    log_scale: bool = False
    default: Any = None

    def sample(self) -> Any:
        """Sample a random value from this dimension."""
        if self.param_type == HyperparameterType.FLOAT:
            if self.log_scale:
                log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
                return np.exp(np.random.uniform(log_low, log_high))
            else:
                return np.random.uniform(self.bounds[0], self.bounds[1])

        elif self.param_type == HyperparameterType.INTEGER:
            if self.log_scale:
                log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
                return int(np.exp(np.random.uniform(log_low, log_high)))
            else:
                return np.random.randint(self.bounds[0], self.bounds[1] + 1)

        elif self.param_type == HyperparameterType.CATEGORICAL:
            return np.random.choice(self.choices)

        elif self.param_type == HyperparameterType.LOG_UNIFORM:
            log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
            return np.exp(np.random.uniform(log_low, log_high))

    def encode(self, value: Any) -> float:
        """Encode parameter value to normalized float for GP."""
        if self.param_type == HyperparameterType.FLOAT:
            if self.log_scale:
                return (np.log(value) - np.log(self.bounds[0])) / (
                    np.log(self.bounds[1]) - np.log(self.bounds[0])
                )
            else:
                return (value - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        elif self.param_type == HyperparameterType.INTEGER:
            if self.log_scale:
                return (np.log(value) - np.log(self.bounds[0])) / (
                    np.log(self.bounds[1]) - np.log(self.bounds[0])
                )
            else:
                return (value - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        elif self.param_type == HyperparameterType.CATEGORICAL:
            return self.choices.index(value) / (len(self.choices) - 1)

        elif self.param_type == HyperparameterType.LOG_UNIFORM:
            return (np.log(value) - np.log(self.bounds[0])) / (
                np.log(self.bounds[1]) - np.log(self.bounds[0])
            )

    def decode(self, encoded_value: float) -> Any:
        """Decode normalized float back to parameter value."""
        if self.param_type == HyperparameterType.FLOAT:
            if self.log_scale:
                log_range = np.log(self.bounds[1]) - np.log(self.bounds[0])
                return np.exp(np.log(self.bounds[0]) + encoded_value * log_range)
            else:
                return self.bounds[0] + encoded_value * (
                    self.bounds[1] - self.bounds[0]
                )

        elif self.param_type == HyperparameterType.INTEGER:
            if self.log_scale:
                log_range = np.log(self.bounds[1]) - np.log(self.bounds[0])
                return int(np.exp(np.log(self.bounds[0]) + encoded_value * log_range))
            else:
                return int(
                    self.bounds[0] + encoded_value * (self.bounds[1] - self.bounds[0])
                )

        elif self.param_type == HyperparameterType.CATEGORICAL:
            index = int(encoded_value * (len(self.choices) - 1))
            return self.choices[min(index, len(self.choices) - 1)]

        elif self.param_type == HyperparameterType.LOG_UNIFORM:
            log_range = np.log(self.bounds[1]) - np.log(self.bounds[0])
            return np.exp(np.log(self.bounds[0]) + encoded_value * log_range)


@dataclass
class HyperparameterSpace:
    """Defines the complete hyperparameter search space."""

    dimensions: List[HyperparameterDimension] = field(default_factory=list)

    def add_dimension(self, dimension: HyperparameterDimension) -> None:
        """Add a hyperparameter dimension to the space."""
        self.dimensions.append(dimension)

    def add_float(
        self, name: str, bounds: Tuple[float, float], log_scale: bool = False
    ) -> None:
        """Add a float hyperparameter dimension."""
        dim = HyperparameterDimension(
            name, HyperparameterType.FLOAT, bounds, log_scale=log_scale
        )
        self.add_dimension(dim)

    def add_integer(
        self, name: str, bounds: Tuple[int, int], log_scale: bool = False
    ) -> None:
        """Add an integer hyperparameter dimension."""
        dim = HyperparameterDimension(
            name, HyperparameterType.INTEGER, bounds, log_scale=log_scale
        )
        self.add_dimension(dim)

    def add_categorical(self, name: str, choices: List[Any]) -> None:
        """Add a categorical hyperparameter dimension."""
        dim = HyperparameterDimension(
            name, HyperparameterType.CATEGORICAL, choices=choices
        )
        self.add_dimension(dim)

    def sample(self) -> Dict[str, Any]:
        """Sample random hyperparameters from the space."""
        return {dim.name: dim.sample() for dim in self.dimensions}

    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameter dict to normalized array for GP."""
        encoded = []
        for dim in self.dimensions:
            if dim.name in params:
                encoded.append(dim.encode(params[dim.name]))
            else:
                encoded.append(0.5)  # Default to middle of range
        return np.array(encoded)

    def decode(self, encoded_params: np.ndarray) -> Dict[str, Any]:
        """Decode normalized array back to parameter dict."""
        params = {}
        for i, dim in enumerate(self.dimensions):
            if i < len(encoded_params):
                params[dim.name] = dim.decode(encoded_params[i])
        return params

    @property
    def dimensionality(self) -> int:
        """Get the dimensionality of the parameter space."""
        return len(self.dimensions)


@dataclass
class OptimizationResult:
    """Stores a hyperparameter evaluation result."""

    participant_id: str
    hyperparameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    round_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participant_id": self.participant_id,
            "hyperparameters": self.hyperparameters,
            "objective_value": self.objective_value,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
        }


class GaussianProcess:
    """Gaussian Process surrogate model for Bayesian optimization."""

    def __init__(self, noise_level: float = 1e-6, length_scale: float = 1.0):
        self.noise_level = noise_level
        self.length_scale = length_scale
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        # Compute squared distances
        X1_expanded = X1[:, np.newaxis, :]  # (n1, 1, d)
        X2_expanded = X2[np.newaxis, :, :]  # (1, n2, d)
        sq_dists = np.sum((X1_expanded - X2_expanded) ** 2, axis=2)

        # RBF kernel
        return np.exp(-0.5 * sq_dists / (self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Compute kernel matrix
        K = self._kernel(self.X_train, self.X_train)
        K += self.noise_level * np.eye(len(self.X_train))

        # Store inverse for predictions
        try:
            self.K_inv = np.linalg.inv(K)
            self.is_fitted = True
        except np.linalg.LinAlgError:
            logger.warning("Kernel matrix is singular, using pseudo-inverse")
            self.K_inv = np.linalg.pinv(K)
            self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points."""
        if not self.is_fitted:
            raise ValueError("GP must be fitted before prediction")

        # Compute kernel between test and training points
        K_star = self._kernel(X, self.X_train)

        # Compute mean
        mean = K_star @ self.K_inv @ self.y_train

        # Compute variance
        K_star_star = self._kernel(X, X)
        var = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
        var = np.maximum(var, 1e-10)  # Ensure numerical stability

        return mean, var


class BayesianOptimizer:
    """Bayesian optimizer with multiple acquisition functions."""

    def __init__(
        self,
        space: HyperparameterSpace,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        exploration_weight: float = 2.0,
        random_seed: Optional[int] = None,
    ):
        self.space = space
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.gp = GaussianProcess()
        self.results = []

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def update(self, results: List[OptimizationResult]) -> None:
        """Update GP with new evaluation results."""
        self.results.extend(results)

        if len(self.results) == 0:
            return

        # Prepare training data
        X = np.array([self.space.encode(r.hyperparameters) for r in self.results])
        y = np.array([r.objective_value for r in self.results])

        # Fit GP
        self.gp.fit(X, y)

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function."""
        if not self.gp.is_fitted or len(self.results) == 0:
            return np.ones(len(X))

        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        # Find best observed value
        best_y = max(r.objective_value for r in self.results)

        # Compute EI
        z = (mean - best_y) / std
        ei = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)

        return ei

    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        if not self.gp.is_fitted:
            return np.ones(len(X))

        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        return mean + self.exploration_weight * std

    def _probability_of_improvement(self, X: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        if not self.gp.is_fitted or len(self.results) == 0:
            return np.ones(len(X))

        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        # Find best observed value
        best_y = max(r.objective_value for r in self.results)

        # Compute PI
        z = (mean - best_y) / std
        pi = norm.cdf(z)

        return pi

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function values."""
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(X)
        elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(X)
        elif (
            self.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT
        ):
            return self._probability_of_improvement(X)
        else:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest next hyperparameters to evaluate."""
        suggestions = []

        for _ in range(n_suggestions):
            if len(self.results) < 3:  # Random initialization
                params = self.space.sample()
            else:
                # Optimize acquisition function
                best_acq = -np.inf
                best_params = None

                # Random search for acquisition function optimization
                for _ in range(1000):  # Random search iterations
                    candidate = np.random.uniform(0, 1, self.space.dimensionality)
                    acq_value = self._acquisition_function(candidate.reshape(1, -1))[0]

                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_params = self.space.decode(candidate)

                params = best_params if best_params else self.space.sample()

            suggestions.append(params)

        return suggestions

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get the best result found so far."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.objective_value)


@dataclass
class FederatedHyperoptConfig:
    """Configuration for federated hyperparameter optimization."""

    max_rounds: int = 20
    suggestions_per_round: int = 3
    min_participants: int = 2
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT
    exploration_weight: float = 2.0
    convergence_tolerance: float = 1e-4
    convergence_window: int = 3
    enable_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5


class FederatedHyperparameterOptimization:
    """Federated hyperparameter optimization coordinator."""

    def __init__(
        self,
        space: HyperparameterSpace,
        config: FederatedHyperoptConfig,
        session_id: Optional[str] = None,
    ):
        self.space = space
        self.config = config
        self.session_id = session_id or f"federated_hyperopt_{int(time.time())}"

        self.optimizer = BayesianOptimizer(
            space=space,
            acquisition_function=config.acquisition_function,
            exploration_weight=config.exploration_weight,
        )

        self.participants = {}
        self.round_history = []
        self.current_round = 0
        self.is_converged = False

        logger.info(
            f"Initialized federated hyperparameter optimization session: {self.session_id}"
        )

    def register_participant(
        self, participant_id: str, capabilities: Dict[str, Any] = None
    ) -> bool:
        """Register a participant for hyperparameter optimization."""
        self.participants[participant_id] = {
            "id": participant_id,
            "capabilities": capabilities or {},
            "active": True,
            "results": [],
            "last_seen": time.time(),
        }

        logger.info(f"Registered participant: {participant_id}")
        return True

    def remove_participant(self, participant_id: str) -> bool:
        """Remove a participant from optimization."""
        if participant_id in self.participants:
            self.participants[participant_id]["active"] = False
            logger.info(f"Removed participant: {participant_id}")
            return True
        return False

    def get_active_participants(self) -> List[str]:
        """Get list of active participants."""
        return [pid for pid, info in self.participants.items() if info["active"]]

    def start_round(self) -> Dict[str, List[Dict[str, Any]]]:
        """Start a new optimization round and distribute suggestions."""
        if self.is_converged:
            logger.info("Optimization has converged, not starting new round")
            return {}

        active_participants = self.get_active_participants()
        if len(active_participants) < self.config.min_participants:
            logger.warning(
                f"Not enough active participants: {len(active_participants)} < {self.config.min_participants}"
            )
            return {}

        self.current_round += 1
        logger.info(f"Starting hyperparameter optimization round {self.current_round}")

        # Generate suggestions for each participant
        participant_suggestions = {}

        for participant_id in active_participants:
            suggestions = self.optimizer.suggest(self.config.suggestions_per_round)
            participant_suggestions[participant_id] = suggestions

            logger.debug(
                f"Generated {len(suggestions)} suggestions for {participant_id}"
            )

        return participant_suggestions

    def collect_results(self, results: List[OptimizationResult]) -> None:
        """Collect and process results from participants."""
        if not results:
            return

        # Update participant records
        for result in results:
            if result.participant_id in self.participants:
                self.participants[result.participant_id]["results"].append(result)
                self.participants[result.participant_id]["last_seen"] = time.time()

        # Apply differential privacy if enabled
        if self.config.enable_privacy:
            results = self._apply_differential_privacy(results)

        # Update optimizer with new results
        self.optimizer.update(results)

        # Record round results
        round_info = {
            "round": self.current_round,
            "num_results": len(results),
            "best_value": max(r.objective_value for r in results) if results else None,
            "participants": list(set(r.participant_id for r in results)),
            "timestamp": time.time(),
        }

        self.round_history.append(round_info)

        # Check for convergence
        self._check_convergence()

        logger.info(f"Collected {len(results)} results for round {self.current_round}")
        if round_info["best_value"] is not None:
            logger.info(f"Best value this round: {round_info['best_value']:.6f}")

    def _apply_differential_privacy(
        self, results: List[OptimizationResult]
    ) -> List[OptimizationResult]:
        """Apply differential privacy to results."""
        if not results:
            return results

        # Add noise to objective values
        sensitivity = 1.0  # Assume normalized objectives
        noise_scale = sensitivity / self.config.dp_epsilon

        private_results = []
        for result in results:
            noise = np.random.laplace(0, noise_scale)
            private_result = OptimizationResult(
                participant_id=result.participant_id,
                hyperparameters=result.hyperparameters,
                objective_value=result.objective_value + noise,
                metrics=result.metrics,
                timestamp=result.timestamp,
                round_number=result.round_number,
            )
            private_results.append(private_result)

        return private_results

    def _check_convergence(self) -> None:
        """Check if optimization has converged."""
        if len(self.round_history) < self.config.convergence_window:
            return

        # Get recent best values
        recent_values = [
            round_info["best_value"]
            for round_info in self.round_history[-self.config.convergence_window :]
            if round_info["best_value"] is not None
        ]

        if len(recent_values) < self.config.convergence_window:
            return

        # Check if improvement is below tolerance
        max_val = max(recent_values)
        min_val = min(recent_values)
        improvement = max_val - min_val

        if improvement < self.config.convergence_tolerance:
            self.is_converged = True
            logger.info(f"Optimization converged after {self.current_round} rounds")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        best_result = self.optimizer.get_best_result()

        return {
            "session_id": self.session_id,
            "current_round": self.current_round,
            "max_rounds": self.config.max_rounds,
            "is_converged": self.is_converged,
            "num_participants": len(self.get_active_participants()),
            "total_evaluations": len(self.optimizer.results),
            "best_result": best_result.to_dict() if best_result else None,
            "round_history": self.round_history[-10:],  # Last 10 rounds
        }

    def export_results(self) -> Dict[str, Any]:
        """Export all optimization results."""
        return {
            "session_id": self.session_id,
            "space": {
                "dimensions": [
                    {
                        "name": dim.name,
                        "type": dim.param_type.value,
                        "bounds": dim.bounds,
                        "choices": dim.choices,
                        "log_scale": dim.log_scale,
                    }
                    for dim in self.space.dimensions
                ]
            },
            "config": {
                "max_rounds": self.config.max_rounds,
                "suggestions_per_round": self.config.suggestions_per_round,
                "acquisition_function": self.config.acquisition_function.value,
                "exploration_weight": self.config.exploration_weight,
            },
            "participants": self.participants,
            "results": [r.to_dict() for r in self.optimizer.results],
            "round_history": self.round_history,
            "optimization_status": self.get_optimization_status(),
        }


# Utility functions for common hyperparameter spaces


def create_neural_network_space() -> HyperparameterSpace:
    """Create a hyperparameter space for neural network optimization."""
    space = HyperparameterSpace()
    space.add_float("learning_rate", (1e-5, 1e-1), log_scale=True)
    space.add_integer("batch_size", (16, 512))
    space.add_integer("hidden_layers", (1, 5))
    space.add_integer("hidden_units", (32, 512))
    space.add_float("dropout_rate", (0.0, 0.8))
    space.add_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    space.add_categorical("activation", ["relu", "tanh", "sigmoid"])
    return space


def create_federated_learning_space() -> HyperparameterSpace:
    """Create a hyperparameter space for federated learning optimization."""
    space = HyperparameterSpace()
    space.add_float("learning_rate", (1e-5, 1e-1), log_scale=True)
    space.add_integer("local_epochs", (1, 10))
    space.add_integer("batch_size", (8, 128))
    space.add_float("client_sampling_rate", (0.1, 1.0))
    space.add_categorical("aggregation_method", ["fedavg", "fedprox", "scaffold"])
    space.add_float("mu", (0.001, 1.0), log_scale=True)  # FedProx regularization
    space.add_integer("communication_rounds", (10, 200))
    return space


def create_xgboost_space() -> HyperparameterSpace:
    """Create a hyperparameter space for XGBoost optimization."""
    space = HyperparameterSpace()
    space.add_integer("n_estimators", (50, 1000))
    space.add_integer("max_depth", (3, 12))
    space.add_float("learning_rate", (0.01, 0.3))
    space.add_float("subsample", (0.5, 1.0))
    space.add_float("colsample_bytree", (0.5, 1.0))
    space.add_float("reg_alpha", (0.0, 10.0))
    space.add_float("reg_lambda", (1.0, 10.0))
    space.add_integer("min_child_weight", (1, 20))
    return space


# Example usage and demo functions


async def demo_federated_hyperparameter_optimization():
    """Demonstrate federated hyperparameter optimization."""
    print("Federated Hyperparameter Optimization Demo")
    print("=" * 50)

    # Create hyperparameter space
    space = create_neural_network_space()

    # Configure optimization
    config = FederatedHyperoptConfig(
        max_rounds=10,
        suggestions_per_round=2,
        min_participants=2,
        acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
    )

    # Initialize federated optimization
    fed_opt = FederatedHyperparameterOptimization(space, config)

    # Register participants
    participants = ["client_1", "client_2", "client_3"]
    for pid in participants:
        fed_opt.register_participant(pid, {"gpu": True, "memory": "8GB"})

    print(f"Registered {len(participants)} participants")

    # Simulate optimization rounds
    for round_num in range(1, config.max_rounds + 1):
        print(f"\nRound {round_num}:")

        # Start round and get suggestions
        suggestions = fed_opt.start_round()

        if not suggestions:
            print("  No suggestions generated (converged or not enough participants)")
            break

        # Simulate participant evaluations
        results = []
        for participant_id, param_sets in suggestions.items():
            for params in param_sets:
                # Simulate objective function (dummy neural network performance)
                score = simulate_objective_function(params)

                result = OptimizationResult(
                    participant_id=participant_id,
                    hyperparameters=params,
                    objective_value=score,
                    round_number=round_num,
                )
                results.append(result)

        # Collect results
        fed_opt.collect_results(results)

        # Show round results
        best_this_round = max(results, key=lambda r: r.objective_value)
        print(f"  Best score this round: {best_this_round.objective_value:.6f}")
        print(f"  Best params: {best_this_round.hyperparameters}")

        if fed_opt.is_converged:
            print("  Optimization converged!")
            break

    # Final results
    print(f"\nOptimization Summary:")
    print("=" * 50)

    status = fed_opt.get_optimization_status()
    print(f"Total rounds: {status['current_round']}")
    print(f"Total evaluations: {status['total_evaluations']}")
    print(f"Converged: {status['is_converged']}")

    if status["best_result"]:
        best = status["best_result"]
        print(f"Best objective value: {best['objective_value']:.6f}")
        print(f"Best hyperparameters: {best['hyperparameters']}")


def simulate_objective_function(params: Dict[str, Any]) -> float:
    """Simulate a noisy objective function for demonstration."""
    # Simulate neural network performance based on hyperparameters
    score = 0.5  # Base score

    # Learning rate contribution (inverted parabola)
    lr = params.get("learning_rate", 0.01)
    score += 0.3 * (1 - abs(np.log10(lr) + 3) / 2)  # Optimal around 1e-3

    # Hidden units contribution
    units = params.get("hidden_units", 128)
    score += 0.2 * min(units / 256, 1.0)  # More units = better (up to a point)

    # Dropout contribution
    dropout = params.get("dropout_rate", 0.2)
    score += 0.1 * (1 - abs(dropout - 0.3))  # Optimal around 0.3

    # Optimizer contribution
    optimizer_scores = {"adam": 0.1, "sgd": 0.05, "rmsprop": 0.08}
    score += optimizer_scores.get(params.get("optimizer", "adam"), 0)

    # Add noise
    score += np.random.normal(0, 0.05)

    return max(0, min(1, score))  # Clamp to [0, 1]


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_federated_hyperparameter_optimization())
