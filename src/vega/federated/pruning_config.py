"""
Configuration settings for Federated Model Pruning
==================================================

This module provides configuration classes and default settings for
federated model pruning and orchestration systems.

Author: Vega2.0 Federated Learning Team
Date: September 2025
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class PruningPreset(Enum):
    """Pre-configured pruning presets for different scenarios."""

    AGGRESSIVE = "aggressive"  # High compression, fast convergence
    BALANCED = "balanced"  # Good balance of compression and accuracy
    CONSERVATIVE = "conservative"  # Safe pruning with minimal accuracy loss
    RESEARCH = "research"  # Experimental settings for research


@dataclass
class FederatedPruningConfig:
    """Main configuration for federated pruning system."""

    # Basic pruning settings
    target_sparsity: float = 0.7
    pruning_method: str = "magnitude"  # magnitude, gradient, structured
    pruning_frequency: int = 2  # Apply pruning every N rounds

    # Knowledge distillation settings
    enable_distillation: bool = True
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    distillation_epochs: int = 5

    # Sparsity scheduling
    initial_sparsity: float = 0.1
    final_sparsity: float = 0.8
    warmup_rounds: int = 3
    cooldown_rounds: int = 5
    adaptation_rate: float = 0.1

    # Performance monitoring
    accuracy_threshold: float = 0.05  # Maximum acceptable accuracy drop
    training_time_threshold: float = 300.0  # Maximum training time (seconds)
    memory_threshold: float = 0.8  # Maximum memory usage

    # Recovery mechanisms
    enable_recovery: bool = True
    recovery_threshold: float = 0.08  # Accuracy drop threshold for recovery
    max_recovery_attempts: int = 3

    # Orchestration settings
    enable_adaptive_orchestration: bool = True
    participant_diversity: bool = True  # Allow different strategies per participant
    dynamic_sparsity: bool = True  # Adapt sparsity based on performance

    def apply_preset(self, preset: PruningPreset):
        """Apply a predefined configuration preset."""
        if preset == PruningPreset.AGGRESSIVE:
            self.target_sparsity = 0.9
            self.initial_sparsity = 0.2
            self.final_sparsity = 0.9
            self.pruning_frequency = 1
            self.adaptation_rate = 0.2
            self.accuracy_threshold = 0.1

        elif preset == PruningPreset.BALANCED:
            self.target_sparsity = 0.7
            self.initial_sparsity = 0.1
            self.final_sparsity = 0.7
            self.pruning_frequency = 2
            self.adaptation_rate = 0.1
            self.accuracy_threshold = 0.05

        elif preset == PruningPreset.CONSERVATIVE:
            self.target_sparsity = 0.5
            self.initial_sparsity = 0.05
            self.final_sparsity = 0.5
            self.pruning_frequency = 3
            self.adaptation_rate = 0.05
            self.accuracy_threshold = 0.03

        elif preset == PruningPreset.RESEARCH:
            self.target_sparsity = 0.8
            self.pruning_method = "structured"
            self.distillation_temperature = 5.0
            self.enable_recovery = True
            self.max_recovery_attempts = 5


@dataclass
class ParticipantConfig:
    """Configuration for individual participants."""

    participant_id: str
    capability_level: str = "medium"  # low, medium, high, variable
    strategy: str = "balanced"  # aggressive, conservative, adaptive, balanced
    max_sparsity: float = 0.8
    min_sparsity: float = 0.1
    computational_budget: float = 1.0
    bandwidth_mbps: float = 100.0
    latency_ms: float = 50.0
    accuracy_tolerance: float = 0.05

    # Hardware constraints
    memory_limit_gb: float = 8.0
    cpu_cores: int = 4
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0


@dataclass
class OrchestrationConfig:
    """Configuration for adaptive orchestration."""

    # Scheduling parameters
    scheduling_algorithm: str = "adaptive"  # adaptive, fixed, performance_based
    round_timeout: float = 600.0  # Maximum time per round (seconds)
    max_concurrent_participants: int = 10

    # Performance monitoring
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, float] = None
    recommendation_system: bool = True

    # History and logging
    save_history: bool = True
    history_file: str = "orchestration_history.json"
    log_level: str = "INFO"
    detailed_metrics: bool = True

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "accuracy_drop": 0.1,
                "training_time": 300.0,
                "memory_usage": 0.8,
                "convergence_rate": 0.01,
            }


class ConfigManager:
    """Manager for federated pruning configurations."""

    DEFAULT_CONFIG_PATH = Path("configs/federated_pruning.yaml")

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> FederatedPruningConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH

        if not config_path.exists():
            # Create default config if it doesn't exist
            config = FederatedPruningConfig()
            cls.save_config(config, config_path)
            return config

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return FederatedPruningConfig(**config_data)

    @classmethod
    def save_config(
        cls, config: FederatedPruningConfig, config_path: Optional[Path] = None
    ):
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False, indent=2)

    @classmethod
    def load_participants(cls, participants_file: Path) -> List[ParticipantConfig]:
        """Load participant configurations from YAML file."""
        if not participants_file.exists():
            return []

        with open(participants_file, "r") as f:
            participants_data = yaml.safe_load(f)

        return [
            ParticipantConfig(**p) for p in participants_data.get("participants", [])
        ]

    @classmethod
    def save_participants(
        cls, participants: List[ParticipantConfig], participants_file: Path
    ):
        """Save participant configurations to YAML file."""
        participants_file.parent.mkdir(parents=True, exist_ok=True)

        participants_data = {"participants": [asdict(p) for p in participants]}

        with open(participants_file, "w") as f:
            yaml.dump(participants_data, f, default_flow_style=False, indent=2)

    @classmethod
    def create_example_config(cls, config_path: Path):
        """Create an example configuration file with comments."""
        config_content = """# Federated Model Pruning Configuration
# =====================================

# Basic pruning settings
target_sparsity: 0.7                    # Target sparsity ratio (0.0 - 0.95)
pruning_method: "magnitude"             # Pruning method: magnitude, gradient, structured
pruning_frequency: 2                    # Apply pruning every N rounds

# Knowledge distillation settings
enable_distillation: true               # Enable knowledge distillation for recovery
distillation_temperature: 4.0           # Temperature for soft targets
distillation_alpha: 0.7                 # Balance between hard and soft targets
distillation_epochs: 5                  # Epochs for distillation training

# Sparsity scheduling
initial_sparsity: 0.1                   # Starting sparsity ratio
final_sparsity: 0.8                     # Final sparsity ratio
warmup_rounds: 3                        # Rounds for gradual sparsity increase
cooldown_rounds: 5                      # Rounds for sparsity stabilization
adaptation_rate: 0.1                    # Rate of adaptation to performance changes

# Performance monitoring
accuracy_threshold: 0.05                # Maximum acceptable accuracy drop
training_time_threshold: 300.0          # Maximum training time in seconds
memory_threshold: 0.8                   # Maximum memory usage ratio

# Recovery mechanisms
enable_recovery: true                   # Enable automatic recovery mechanisms
recovery_threshold: 0.08                # Accuracy drop threshold for recovery
max_recovery_attempts: 3                # Maximum recovery attempts per participant

# Orchestration settings
enable_adaptive_orchestration: true     # Enable intelligent orchestration
participant_diversity: true             # Allow different strategies per participant
dynamic_sparsity: true                  # Adapt sparsity based on real-time performance
"""

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_content)

    @classmethod
    def create_example_participants(cls, participants_file: Path):
        """Create an example participants configuration file."""
        participants_content = """# Federated Pruning Participants Configuration
# ===========================================

participants:
  - participant_id: "mobile_device_1"
    capability_level: "low"
    strategy: "conservative"
    max_sparsity: 0.4
    min_sparsity: 0.05
    computational_budget: 0.3
    bandwidth_mbps: 20.0
    latency_ms: 100.0
    accuracy_tolerance: 0.03
    memory_limit_gb: 2.0
    cpu_cores: 2
    has_gpu: false
    gpu_memory_gb: 0.0
  
  - participant_id: "edge_server_1"
    capability_level: "medium"
    strategy: "balanced"
    max_sparsity: 0.7
    min_sparsity: 0.1
    computational_budget: 1.0
    bandwidth_mbps: 100.0
    latency_ms: 50.0
    accuracy_tolerance: 0.05
    memory_limit_gb: 8.0
    cpu_cores: 8
    has_gpu: true
    gpu_memory_gb: 4.0
  
  - participant_id: "cloud_instance_1"
    capability_level: "high"
    strategy: "aggressive"
    max_sparsity: 0.95
    min_sparsity: 0.1
    computational_budget: 2.0
    bandwidth_mbps: 1000.0
    latency_ms: 20.0
    accuracy_tolerance: 0.08
    memory_limit_gb: 32.0
    cpu_cores: 16
    has_gpu: true
    gpu_memory_gb: 16.0
  
  - participant_id: "variable_device_1"
    capability_level: "variable"
    strategy: "adaptive"
    max_sparsity: 0.8
    min_sparsity: 0.1
    computational_budget: 1.5
    bandwidth_mbps: 200.0
    latency_ms: 30.0
    accuracy_tolerance: 0.06
    memory_limit_gb: 16.0
    cpu_cores: 12
    has_gpu: true
    gpu_memory_gb: 8.0
"""

        participants_file.parent.mkdir(parents=True, exist_ok=True)
        with open(participants_file, "w") as f:
            f.write(participants_content)


def validate_config(config: FederatedPruningConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Validate sparsity values
    if not 0.0 <= config.target_sparsity <= 0.95:
        issues.append("target_sparsity must be between 0.0 and 0.95")

    if not 0.0 <= config.initial_sparsity <= config.final_sparsity <= 0.95:
        issues.append("sparsity progression must be: 0.0 ≤ initial ≤ final ≤ 0.95")

    if config.initial_sparsity > config.target_sparsity:
        issues.append("initial_sparsity cannot exceed target_sparsity")

    # Validate distillation parameters
    if config.distillation_temperature <= 0:
        issues.append("distillation_temperature must be positive")

    if not 0.0 <= config.distillation_alpha <= 1.0:
        issues.append("distillation_alpha must be between 0.0 and 1.0")

    # Validate thresholds
    if config.accuracy_threshold < 0:
        issues.append("accuracy_threshold must be non-negative")

    if config.training_time_threshold <= 0:
        issues.append("training_time_threshold must be positive")

    if not 0.0 <= config.memory_threshold <= 1.0:
        issues.append("memory_threshold must be between 0.0 and 1.0")

    # Validate orchestration settings
    if config.adaptation_rate < 0:
        issues.append("adaptation_rate must be non-negative")

    return issues


# Example usage and defaults
def get_default_config() -> FederatedPruningConfig:
    """Get default federated pruning configuration."""
    return FederatedPruningConfig()


def get_preset_config(preset: PruningPreset) -> FederatedPruningConfig:
    """Get configuration with a specific preset applied."""
    config = FederatedPruningConfig()
    config.apply_preset(preset)
    return config


if __name__ == "__main__":
    # Example: Create default configuration files
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Create example configuration
    config_manager = ConfigManager()
    config_manager.create_example_config(config_dir / "federated_pruning_example.yaml")
    config_manager.create_example_participants(config_dir / "participants_example.yaml")

    print("✅ Created example configuration files:")
    print(f"   - {config_dir}/federated_pruning_example.yaml")
    print(f"   - {config_dir}/participants_example.yaml")

    # Test configuration loading and validation
    config = get_default_config()
    issues = validate_config(config)

    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Default configuration is valid")

    # Test presets
    for preset in PruningPreset:
        preset_config = get_preset_config(preset)
        preset_issues = validate_config(preset_config)
        print(
            f"✅ {preset.value} preset: {'valid' if not preset_issues else 'has issues'}"
        )
