"""
Federated learning algorithms and implementations for Vega2.0.

This module provides advanced federated learning capabilities including
meta-learning and Byzantine-robust approaches.
"""

from .meta_learning import (
    FederatedMAML,
    MAMLConfig,
    Task,
    SimpleMetaModel,
)

from .byzantine_robust import (
    ByzantineConfig,
    ParticipantUpdate,
    SimpleByzantineModel,
    ByzantineRobustAggregator,
    ByzantineAttackSimulator,
    run_byzantine_robust_fl,
)

__all__ = [
    # Meta-learning
    "FederatedMAML",
    "MAMLConfig",
    "Task",
    "SimpleMetaModel",
    # Byzantine-robust FL
    "ByzantineConfig",
    "ParticipantUpdate",
    "SimpleByzantineModel",
    "ByzantineRobustAggregator",
    "ByzantineAttackSimulator",
    "run_byzantine_robust_fl",
]

from .model_serialization import ModelWeights, ModelSerializer
from .coordinator import FederatedCoordinator
from .participant import FederatedParticipant
from .encryption import DynamicEncryption
from .fedavg import FedAvg, FedAvgConfig
from .algorithms import FedProx, FedProxConfig, LAG, LAGConfig
from .dp import DifferentialPrivacy
from .image_input import (
    ImageInputConfig,
    ImageInputHandler,
    ProcessedImage,
    process_image,
)
from .data_utils import (
    DataStatistics,
    PartitionResult,
    partition_dataset,
    compute_data_statistics,
)
from .cross_silo import (
    Organization,
    Silo,
    HierarchicalParticipant,
    OrganizationRole,
    FederationLevel,
    OrganizationManager,
)
from .cross_silo_coordinator import CrossSiloCoordinator, CrossSiloCoordinationConfig
from .hierarchical_aggregation import HierarchicalAggregator, LevelAggregationConfig
from .reinforcement import (
    BanditEnv,
    SoftmaxPolicy,
    LocalFRLConfig,
    local_train_bandit,
    fedavg_thetas,
    run_federated_bandit,
)
from .continual import (
    Task,
    LinearModel,
    EWCConfig,
    EWCParticipant,
    federated_continual_aggregate,
    run_continual_federated_learning,
    create_synthetic_task_sequence,
)
from .async_fl import (
    AsyncUpdate,
    ParticipantState,
    AsyncFLConfig,
    SimpleAsyncModel,
    AsyncParticipant,
    AsyncFLCoordinator,
    run_async_federated_learning,
    generate_heterogeneous_async_data,
)
from .meta_learning import (
    Task,
    MAMLConfig,
    SimpleMetaModel,
    FederatedMAML,
    generate_sine_wave_tasks,
    run_federated_maml,
)
from .hyperopt import (
    HyperparameterType,
    AcquisitionFunction,
    HyperparameterDimension,
    HyperparameterSpace,
    OptimizationResult,
    GaussianProcess,
    BayesianOptimizer,
    FederatedHyperoptConfig,
    FederatedHyperparameterOptimization,
    create_neural_network_space,
    create_federated_learning_space,
    create_xgboost_space,
)

__all__ = [
    # Core components
    "ModelWeights",
    "ModelSerializer",
    "FederatedCoordinator",
    "FederatedParticipant",
    "DynamicEncryption",
    # Aggregation algorithms
    "FedAvg",
    "FedAvgConfig",
    "FedProx",
    "FedProxConfig",
    "LAG",
    "LAGConfig",
    # Privacy and security
    "DifferentialPrivacy",
    "ImageInputConfig",
    "ImageInputHandler",
    "ProcessedImage",
    "process_image",
    # Data utilities
    "DataStatistics",
    "PartitionResult",
    "partition_dataset",
    "compute_data_statistics",
    # Cross-silo federation
    "Organization",
    "Silo",
    "HierarchicalParticipant",
    "OrganizationRole",
    "FederationLevel",
    "OrganizationManager",
    "CrossSiloCoordinator",
    "CrossSiloCoordinationConfig",
    "HierarchicalAggregator",
    "LevelAggregationConfig",
    # Federated Reinforcement Learning (Bandit)
    "BanditEnv",
    "SoftmaxPolicy",
    "LocalFRLConfig",
    "local_train_bandit",
    "fedavg_thetas",
    "run_federated_bandit",
    # Continual Federated Learning (EWC)
    "Task",
    "LinearModel",
    "EWCConfig",
    "EWCParticipant",
    "federated_continual_aggregate",
    "run_continual_federated_learning",
    "create_synthetic_task_sequence",
    # Asynchronous Federated Learning
    "AsyncUpdate",
    "ParticipantState",
    "AsyncFLConfig",
    "SimpleAsyncModel",
    "AsyncParticipant",
    "AsyncFLCoordinator",
    "run_async_federated_learning",
    "generate_heterogeneous_async_data",
    # Federated Meta-Learning (MAML)
    "MAMLConfig",
    "SimpleMetaModel",
    "FederatedMAML",
    "generate_sine_wave_tasks",
    "run_federated_maml",
    # Federated Hyperparameter Optimization
    "HyperparameterType",
    "AcquisitionFunction",
    "HyperparameterDimension",
    "HyperparameterSpace",
    "OptimizationResult",
    "GaussianProcess",
    "BayesianOptimizer",
    "FederatedHyperoptConfig",
    "FederatedHyperparameterOptimization",
    "create_neural_network_space",
    "create_federated_learning_space",
    "create_xgboost_space",
]

__version__ = "1.0.0"
__author__ = "Vega 2.0 Team"
