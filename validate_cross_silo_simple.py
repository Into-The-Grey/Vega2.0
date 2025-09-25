"""
Simple Cross-Silo Hierarchical Federated Learning Validation.

Tests core cross-silo concepts and hierarchical aggregation without
complex dependencies, focusing on organizational structure and federation logic.
"""

import random
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PrivacyLevel(Enum):
    """Privacy levels for organizations."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    MAXIMUM = "MAXIMUM"


class OrganizationRole(Enum):
    """Roles within an organization."""

    ADMIN = "admin"
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


@dataclass
class OrganizationConfig:
    """Configuration for an organization in cross-silo federation."""

    org_id: str
    name: str
    domain: str
    privacy_level: str
    max_participants: int = 50
    min_participants: int = 3
    data_retention_days: int = 90
    audit_enabled: bool = True
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5


@dataclass
class Participant:
    """Represents a participant within an organization."""

    participant_id: str
    role: OrganizationRole
    org_id: str
    metadata: Dict[str, Any]
    is_active: bool = True


class Organization:
    """Represents an organization in cross-silo federation."""

    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.participants: List[Participant] = []
        self.local_model: Optional[Dict[str, Any]] = None
        self.aggregation_history: List[Dict[str, Any]] = []

    def add_participant(
        self,
        participant_id: str,
        role: OrganizationRole,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a participant to the organization."""
        if len(self.participants) >= self.config.max_participants:
            return False

        participant = Participant(
            participant_id=participant_id,
            role=role,
            org_id=self.config.org_id,
            metadata=metadata or {},
        )

        self.participants.append(participant)
        return True

    def aggregate_local_models(
        self, participant_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate participant updates within the organization."""
        if not participant_updates:
            return {}

        # Simple federated averaging within organization
        aggregated = {}

        # Get structure from first update
        first_update = participant_updates[0]

        # Skip non-numeric fields that shouldn't be aggregated
        skip_fields = {"participant_id", "round"}

        for key in first_update:
            if key in skip_fields:
                continue

            if isinstance(first_update[key], list):
                if isinstance(first_update[key][0], list):
                    # 2D parameter matrix
                    rows = len(first_update[key])
                    cols = len(first_update[key][0])
                    aggregated[key] = []

                    for i in range(rows):
                        aggregated[key].append([])
                        for j in range(cols):
                            values = [
                                update[key][i][j] for update in participant_updates
                            ]
                            aggregated[key][i].append(sum(values) / len(values))
                else:
                    # 1D parameter vector
                    aggregated[key] = []
                    for i in range(len(first_update[key])):
                        values = [update[key][i] for update in participant_updates]
                        aggregated[key].append(sum(values) / len(values))
            else:
                # Scalar value (numeric only)
                try:
                    values = [update[key] for update in participant_updates]
                    aggregated[key] = sum(values) / len(values)
                except (TypeError, ValueError):
                    # Skip non-numeric scalar values
                    continue

        # Apply differential privacy if enabled
        if self.config.differential_privacy:
            aggregated = self._apply_differential_privacy(aggregated)

        self.local_model = aggregated
        return aggregated

    def _apply_differential_privacy(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy to model parameters."""
        dp_model = {}
        noise_scale = 2.0 / self.config.dp_epsilon  # Simplified DP noise

        for key, value in model.items():
            if isinstance(value, list):
                if isinstance(value[0], list):
                    # 2D matrix
                    dp_model[key] = []
                    for i in range(len(value)):
                        dp_model[key].append([])
                        for j in range(len(value[i])):
                            noise = random.gauss(0, noise_scale * 0.01)  # Small noise
                            dp_model[key][i].append(value[i][j] + noise)
                else:
                    # 1D vector
                    dp_model[key] = []
                    for i in range(len(value)):
                        noise = random.gauss(0, noise_scale * 0.01)
                        dp_model[key].append(value[i] + noise)
            else:
                # Scalar
                noise = random.gauss(0, noise_scale * 0.01)
                dp_model[key] = value + noise

        return dp_model


@dataclass
class CrossSiloConfig:
    """Configuration for cross-silo federated learning."""

    global_rounds: int = 10
    org_aggregation_strategy: str = "fedavg"
    global_aggregation_strategy: str = "weighted_fedavg"
    min_organizations: int = 2
    max_staleness_rounds: int = 3
    convergence_threshold: float = 0.01
    differential_privacy_enabled: bool = False
    audit_logging: bool = True
    secure_aggregation: bool = False
    communication_encryption: bool = False


class HierarchicalAggregator:
    """Hierarchical aggregator for cross-silo federation."""

    def __init__(self, config: CrossSiloConfig):
        self.config = config

    def aggregate_cross_silo(
        self, org_contributions: Dict[str, Dict[str, Any]], round_number: int
    ) -> Dict[str, Any]:
        """Aggregate organizational model contributions into global model."""
        if not org_contributions:
            return {}

        # Weighted aggregation based on participant count
        org_weights = {}
        total_participants = 0

        for org_id, contribution in org_contributions.items():
            participant_count = contribution.get("metadata", {}).get(
                "participant_count", 1
            )
            org_weights[org_id] = participant_count
            total_participants += participant_count

        # Normalize weights
        for org_id in org_weights:
            org_weights[org_id] = org_weights[org_id] / total_participants

        # Weighted average of organizational models
        global_model = {}
        first_contribution = list(org_contributions.values())[0]

        for key in first_contribution:
            if key == "metadata":
                continue  # Skip metadata in aggregation

            if isinstance(first_contribution[key], list):
                if isinstance(first_contribution[key][0], list):
                    # 2D parameter matrix
                    rows = len(first_contribution[key])
                    cols = len(first_contribution[key][0])
                    global_model[key] = []

                    for i in range(rows):
                        global_model[key].append([])
                        for j in range(cols):
                            weighted_sum = 0.0
                            for org_id, contribution in org_contributions.items():
                                weight = org_weights[org_id]
                                weighted_sum += weight * contribution[key][i][j]
                            global_model[key][i].append(weighted_sum)
                else:
                    # 1D parameter vector
                    global_model[key] = []
                    for i in range(len(first_contribution[key])):
                        weighted_sum = 0.0
                        for org_id, contribution in org_contributions.items():
                            weight = org_weights[org_id]
                            weighted_sum += weight * contribution[key][i]
                        global_model[key].append(weighted_sum)
            else:
                # Scalar value
                weighted_sum = 0.0
                for org_id, contribution in org_contributions.items():
                    weight = org_weights[org_id]
                    weighted_sum += weight * contribution[key]
                global_model[key] = weighted_sum

        return global_model

    def validate_privacy_preservation(
        self, org_contributions: Dict[str, Dict[str, Any]], global_model: Dict[str, Any]
    ) -> bool:
        """Validate that privacy is preserved in aggregation."""
        # Check if any organization with HIGH/MAXIMUM privacy contributed
        high_privacy_orgs = []
        for org_id, contribution in org_contributions.items():
            privacy_level = contribution.get("metadata", {}).get("privacy_level", "LOW")
            if privacy_level in ["HIGH", "MAXIMUM"]:
                high_privacy_orgs.append(org_id)

        # For high privacy organizations, ensure individual contributions
        # cannot be reverse-engineered from global model
        if high_privacy_orgs and len(org_contributions) < 3:
            return False  # Need at least 3 orgs for privacy

        return True


class CrossSiloFederatedLearning:
    """Main cross-silo federated learning system."""

    def __init__(self, config: CrossSiloConfig, organizations: List[Organization]):
        self.config = config
        self.organizations = organizations
        self.global_model: Optional[Dict[str, Any]] = None
        self.round_history: List[Dict[str, Any]] = []
        self.aggregator = HierarchicalAggregator(config)

    def create_session(
        self,
        session_id: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> "CrossSiloSession":
        """Create a new cross-silo federation session."""
        return CrossSiloSession(
            session_id=session_id,
            cross_silo_fl=self,
            model_config=model_config,
            training_config=training_config,
        )

    def run_federated_round(self, round_number: int) -> Dict[str, Any]:
        """Run a single round of cross-silo federated learning."""
        # Collect organizational contributions
        org_contributions = {}

        for org in self.organizations:
            if len(org.participants) >= org.config.min_participants:
                try:
                    # Simulate organizational aggregation
                    participant_updates = self._simulate_participant_updates(
                        org, round_number
                    )
                    org_model = org.aggregate_local_models(participant_updates)

                    if org_model:
                        org_contributions[org.config.org_id] = {
                            **org_model,
                            "metadata": {
                                "org_id": org.config.org_id,
                                "participant_count": len(org.participants),
                                "privacy_level": org.config.privacy_level,
                                "round_number": round_number,
                            },
                        }
                except Exception as e:
                    print(f"  Error in organization {org.config.org_id}: {e}")
                    continue

        # Global aggregation
        if len(org_contributions) >= self.config.min_organizations:
            try:
                self.global_model = self.aggregator.aggregate_cross_silo(
                    org_contributions, round_number
                )

                # Record round history
                round_result = {
                    "round": round_number,
                    "participating_orgs": list(org_contributions.keys()),
                    "total_participants": sum(
                        contrib["metadata"]["participant_count"]
                        for contrib in org_contributions.values()
                    ),
                    "privacy_preserved": self.aggregator.validate_privacy_preservation(
                        org_contributions, self.global_model
                    ),
                }

                self.round_history.append(round_result)
                return round_result
            except Exception as e:
                raise ValueError(f"Global aggregation failed: {e}")
        else:
            raise ValueError(
                f"Insufficient organizations: {len(org_contributions)} < {self.config.min_organizations}"
            )

    def _simulate_participant_updates(
        self, org: Organization, round_number: int
    ) -> List[Dict[str, Any]]:
        """Simulate participant model updates within an organization."""
        updates = []

        for i, participant in enumerate(org.participants):
            if not participant.is_active:
                continue

            # Generate mock model update
            update = {
                "weights": [[random.gauss(0, 0.1) for _ in range(5)] for _ in range(3)],
                "bias": [random.gauss(0, 0.05) for _ in range(5)],
                "loss": random.uniform(0.1, 2.0),
                "participant_id": participant.participant_id,
                "round": round_number,
            }

            updates.append(update)

        return updates


class CrossSiloSession:
    """Represents an active cross-silo federation session."""

    def __init__(
        self,
        session_id: str,
        cross_silo_fl: CrossSiloFederatedLearning,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        self.session_id = session_id
        self.cross_silo_fl = cross_silo_fl
        self.model_config = model_config
        self.training_config = training_config
        self.status = "ACTIVE"
        self.participating_orgs = [
            org.config.org_id for org in cross_silo_fl.organizations
        ]


def demo() -> None:
    """Run cross-silo hierarchical federated learning demonstration."""
    print("Cross-Silo Hierarchical Federated Learning Demo")
    print("=" * 60)

    # Create organizations with different privacy levels
    org_configs = [
        OrganizationConfig(
            org_id="healthcare_corp",
            name="Healthcare Corp",
            domain="healthcare.example.com",
            privacy_level="HIGH",
            min_participants=3,
            max_participants=20,
            differential_privacy=True,
            dp_epsilon=1.0,
        ),
        OrganizationConfig(
            org_id="financial_services",
            name="Financial Services",
            domain="finance.example.com",
            privacy_level="MAXIMUM",
            min_participants=5,
            max_participants=15,
            differential_privacy=True,
            dp_epsilon=0.5,
        ),
        OrganizationConfig(
            org_id="research_university",
            name="Research University",
            domain="research.edu",
            privacy_level="MEDIUM",
            min_participants=2,
            max_participants=30,
            differential_privacy=False,
        ),
    ]

    organizations = [Organization(config) for config in org_configs]

    # Add participants to organizations
    participant_counts = [8, 12, 15]  # Different sizes

    for org, count in zip(organizations, participant_counts):
        for i in range(count):
            participant_id = f"participant_{org.config.org_id}_{i}"
            org.add_participant(participant_id, OrganizationRole.PARTICIPANT)

    print(f"Created {len(organizations)} organizations:")
    for org in organizations:
        print(
            f"  - {org.config.name}: {len(org.participants)} participants, "
            f"Privacy: {org.config.privacy_level}, DP: {org.config.differential_privacy}"
        )

    # Configure cross-silo federation
    config = CrossSiloConfig(
        global_rounds=5,
        min_organizations=2,
        differential_privacy_enabled=True,
        audit_logging=True,
    )

    cross_silo_fl = CrossSiloFederatedLearning(config, organizations)

    # Run federation rounds
    print(f"\nRunning {config.global_rounds} rounds of cross-silo federation...")

    for round_num in range(config.global_rounds):
        try:
            result = cross_silo_fl.run_federated_round(round_num + 1)

            print(f"\nRound {result['round']}:")
            print(
                f"  - Participating organizations: {len(result['participating_orgs'])}"
            )
            print(f"  - Total participants: {result['total_participants']}")
            print(f"  - Privacy preserved: {result['privacy_preserved']}")
            print(f"  - Organizations: {', '.join(result['participating_orgs'])}")

        except Exception as e:
            print(f"\nRound {round_num + 1} failed: {e}")

    # Summary
    print(f"\n" + "=" * 60)
    print("Cross-Silo Federation Summary")
    print("=" * 60)

    total_rounds = len(cross_silo_fl.round_history)
    total_participants = sum(len(org.participants) for org in organizations)
    privacy_orgs = sum(1 for org in organizations if org.config.differential_privacy)

    print(f"✓ Completed rounds: {total_rounds}")
    print(f"✓ Organizations: {len(organizations)}")
    print(f"✓ Total participants: {total_participants}")
    print(f"✓ Privacy-enabled orgs: {privacy_orgs}")
    print(f"✓ Global model created: {cross_silo_fl.global_model is not None}")

    if cross_silo_fl.round_history:
        privacy_preserved = all(
            r["privacy_preserved"] for r in cross_silo_fl.round_history
        )
        print(f"✓ Privacy preserved: {privacy_preserved}")

    print(f"\nCross-silo features demonstrated:")
    print(f"  - Multi-organizational hierarchy")
    print(f"  - Organization-level privacy controls")
    print(f"  - Differential privacy integration")
    print(f"  - Hierarchical aggregation (Org → Global)")
    print(f"  - Weighted federation by participant count")
    print(f"  - Privacy preservation validation")


if __name__ == "__main__":
    demo()
