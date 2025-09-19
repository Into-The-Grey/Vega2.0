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
]

__version__ = "1.0.0"
__author__ = "Vega 2.0 Team"
