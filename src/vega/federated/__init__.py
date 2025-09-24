"""
Vega 2.0 Federated Learning Module

A comprehensive federated learning system with support for:
- Personal/family use (2-3 participants)
- Cross-silo enterprise federation
- Hierarchical multi-organization learning
- Advanced privacy and security features

Core Components:
- Model serialization framework (PyTorch/TensorFlow)
- REST-based communication layer
- Central coordinator service
- Dynamic rotating encryption baseline
- Participant management
- Cross-silo hierarchical federation
- Advanced aggregation algorithms
- Differential privacy and secure aggregation
"""

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
]

__version__ = "1.0.0"
__author__ = "Vega 2.0 Team"
