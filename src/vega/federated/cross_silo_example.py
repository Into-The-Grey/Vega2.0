"""
Cross-Silo Federated Learning Example

Demonstrates the complete cross-silo federated learning workflow including:
1. Organization and silo setup
2. Participant registration
3. Cross-silo session creation
4. Hierarchical federated learning execution

This example simulates a healthcare federation with multiple hospitals
(organizations) each having different departments (silos) with participating
research teams (participants).
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our cross-silo federated learning components
from cross_silo import OrganizationRole, FederationLevel
from cross_silo_coordinator import CrossSiloCoordinator, CrossSiloCoordinationConfig
from hierarchical_aggregation import LevelAggregationConfig
from model_serialization import ModelWeights


class CrossSiloFederatedLearningDemo:
    """Comprehensive demo of cross-silo federated learning."""

    def __init__(self):
        """Initialize the demo."""
        # Configure coordinator for healthcare scenario
        config = CrossSiloCoordinationConfig(
            min_organizations=2,
            min_silos_per_org=2,
            min_participants_per_silo=2,
            max_rounds=10,
            convergence_threshold=0.01,
            silo_aggregation_timeout=180.0,
            org_aggregation_timeout=300.0,
            global_aggregation_timeout=600.0,
            enable_domain_adaptation=True,
            model_validation_strict=True,
        )

        self.coordinator = CrossSiloCoordinator(config)

        # Demo data
        self.organizations = {}
        self.silos = {}
        self.participants = {}
        self.session = None

        logger.info("Cross-Silo Federated Learning Demo initialized")

    async def setup_healthcare_federation(self):
        """Setup a healthcare federation with multiple hospitals."""
        logger.info("Setting up healthcare federation...")

        # Create three healthcare organizations
        hospital_configs = [
            {
                "name": "Metro General Hospital",
                "description": "Large urban teaching hospital with comprehensive research programs",
                "domain": "healthcare",
                "admin_contact": "research@metro-general.org",
                "min_participants": 2,
                "max_participants": 50,
                "data_sharing_policy": "strict",
            },
            {
                "name": "Regional Medical Center",
                "description": "Regional hospital serving rural communities",
                "domain": "healthcare",
                "admin_contact": "data@regional-med.org",
                "min_participants": 2,
                "max_participants": 30,
                "data_sharing_policy": "moderate",
            },
            {
                "name": "Children's Hospital Network",
                "description": "Specialized pediatric hospital network",
                "domain": "healthcare",
                "admin_contact": "research@childrens-net.org",
                "min_participants": 1,
                "max_participants": 25,
                "data_sharing_policy": "strict",
            },
        ]

        # Create organizations
        for config in hospital_configs:
            org = await self.coordinator.org_manager.create_organization(**config)
            self.organizations[org.org_id] = org
            logger.info(f"Created organization: {org.name} ({org.org_id})")

        # Create silos (departments) within each organization
        await self._create_hospital_departments()

        # Register participants (research teams) in each silo
        await self._register_research_teams()

        logger.info("Healthcare federation setup completed!")
        return self.organizations

    async def _create_hospital_departments(self):
        """Create departments (silos) within each hospital."""
        # Metro General Hospital departments
        metro_org_id = list(self.organizations.keys())[0]
        metro_departments = [
            {
                "name": "Cardiology Research Unit",
                "description": "Cardiovascular disease research with ECG and imaging data",
                "data_type": "medical_imaging",
                "location": "Metro General - Building A, Floor 3",
                "contact": "cardio-research@metro-general.org",
                "data_size": 15000,
                "compute_capacity": "high",
            },
            {
                "name": "Oncology Data Science Lab",
                "description": "Cancer research with genomic and clinical data",
                "data_type": "genomic",
                "location": "Metro General - Research Center",
                "contact": "onco-data@metro-general.org",
                "data_size": 8000,
                "compute_capacity": "high",
            },
            {
                "name": "Emergency Medicine Analytics",
                "description": "Emergency department outcome prediction models",
                "data_type": "tabular",
                "location": "Metro General - ED Research Office",
                "contact": "ed-analytics@metro-general.org",
                "data_size": 25000,
                "compute_capacity": "medium",
            },
        ]

        for dept_config in metro_departments:
            silo = await self.coordinator.org_manager.create_silo(
                org_id=metro_org_id, **dept_config
            )
            self.silos[silo.silo_id] = silo
            logger.info(f"Created silo: {silo.name} ({silo.silo_id})")

        # Regional Medical Center departments
        regional_org_id = list(self.organizations.keys())[1]
        regional_departments = [
            {
                "name": "Primary Care Research",
                "description": "Community health and preventive care studies",
                "data_type": "tabular",
                "location": "Regional Med - Outpatient Research",
                "contact": "primary-care@regional-med.org",
                "data_size": 12000,
                "compute_capacity": "medium",
            },
            {
                "name": "Rural Health Informatics",
                "description": "Telemedicine and rural health outcomes",
                "data_type": "mixed",
                "location": "Regional Med - IT Department",
                "contact": "health-informatics@regional-med.org",
                "data_size": 7500,
                "compute_capacity": "medium",
            },
        ]

        for dept_config in regional_departments:
            silo = await self.coordinator.org_manager.create_silo(
                org_id=regional_org_id, **dept_config
            )
            self.silos[silo.silo_id] = silo
            logger.info(f"Created silo: {silo.name} ({silo.silo_id})")

        # Children's Hospital departments
        children_org_id = list(self.organizations.keys())[2]
        children_departments = [
            {
                "name": "Pediatric Cardiology",
                "description": "Congenital heart disease research",
                "data_type": "medical_imaging",
                "location": "Children's Hospital - Cardiac Center",
                "contact": "ped-cardio@childrens-net.org",
                "data_size": 5000,
                "compute_capacity": "medium",
            },
            {
                "name": "Developmental Medicine",
                "description": "Child development and neurological studies",
                "data_type": "tabular",
                "location": "Children's Hospital - Development Center",
                "contact": "dev-med@childrens-net.org",
                "data_size": 9000,
                "compute_capacity": "medium",
            },
        ]

        for dept_config in children_departments:
            silo = await self.coordinator.org_manager.create_silo(
                org_id=children_org_id, **dept_config
            )
            self.silos[silo.silo_id] = silo
            logger.info(f"Created silo: {silo.name} ({silo.silo_id})")

    async def _register_research_teams(self):
        """Register research teams as participants in each silo."""
        # Metro General participants
        metro_silos = [
            s
            for s in self.silos.values()
            if s.org_id == list(self.organizations.keys())[0]
        ]

        for silo in metro_silos:
            # Each silo gets 2-3 research teams
            team_count = 3 if "Cardiology" in silo.name else 2

            for i in range(team_count):
                participant = await self.coordinator.org_manager.register_participant(
                    org_id=silo.org_id,
                    silo_id=silo.silo_id,
                    name=f"{silo.name} - Research Team {i+1}",
                    role=OrganizationRole.PARTICIPANT,
                    contact=f"team{i+1}@{silo.contact}",
                    data_characteristics={
                        "sample_count": silo.data_size // team_count,
                        "data_quality": "high",
                        "annotation_completeness": 0.95,
                    },
                    compute_resources={
                        "gpu_count": 2 if silo.compute_capacity == "high" else 1,
                        "memory_gb": 64,
                        "relative_power": 1.0,
                    },
                    performance_metrics={
                        "accuracy": 0.85 + np.random.random() * 0.1,
                        "reliability": 0.95,
                    },
                )
                self.participants[participant.participant_id] = participant
                logger.info(f"Registered participant: {participant.name}")

        # Regional Medical participants
        regional_silos = [
            s
            for s in self.silos.values()
            if s.org_id == list(self.organizations.keys())[1]
        ]

        for silo in regional_silos:
            for i in range(2):
                participant = await self.coordinator.org_manager.register_participant(
                    org_id=silo.org_id,
                    silo_id=silo.silo_id,
                    name=f"{silo.name} - Team {i+1}",
                    role=OrganizationRole.PARTICIPANT,
                    contact=f"team{i+1}@{silo.contact}",
                    data_characteristics={
                        "sample_count": silo.data_size // 2,
                        "data_quality": "medium",
                        "annotation_completeness": 0.88,
                    },
                    compute_resources={
                        "gpu_count": 1,
                        "memory_gb": 32,
                        "relative_power": 0.8,
                    },
                    performance_metrics={
                        "accuracy": 0.80 + np.random.random() * 0.1,
                        "reliability": 0.90,
                    },
                )
                self.participants[participant.participant_id] = participant
                logger.info(f"Registered participant: {participant.name}")

        # Children's Hospital participants
        children_silos = [
            s
            for s in self.silos.values()
            if s.org_id == list(self.organizations.keys())[2]
        ]

        for silo in children_silos:
            for i in range(2):
                participant = await self.coordinator.org_manager.register_participant(
                    org_id=silo.org_id,
                    silo_id=silo.silo_id,
                    name=f"{silo.name} - Research Group {i+1}",
                    role=OrganizationRole.PARTICIPANT,
                    contact=f"group{i+1}@{silo.contact}",
                    data_characteristics={
                        "sample_count": silo.data_size // 2,
                        "data_quality": "high",
                        "annotation_completeness": 0.92,
                    },
                    compute_resources={
                        "gpu_count": 1,
                        "memory_gb": 48,
                        "relative_power": 0.9,
                    },
                    performance_metrics={
                        "accuracy": 0.82 + np.random.random() * 0.1,
                        "reliability": 0.93,
                    },
                )
                self.participants[participant.participant_id] = participant
                logger.info(f"Registered participant: {participant.name}")

    async def create_federated_session(self):
        """Create a cross-silo federated learning session."""
        logger.info("Creating cross-silo federated learning session...")

        # Setup participating organizations and silos
        participating_orgs = list(self.organizations.keys())
        participating_silos = {}

        for org_id in participating_orgs:
            org_silos = [s.silo_id for s in self.silos.values() if s.org_id == org_id]
            participating_silos[org_id] = org_silos

        # Create session
        self.session = await self.coordinator.create_cross_silo_session(
            name="Multi-Hospital Heart Disease Prediction",
            description="Federated learning for cardiovascular risk prediction across multiple healthcare organizations",
            participating_orgs=participating_orgs,
            participating_silos=participating_silos,
            federation_topology="star",
            domain_adaptation=True,
            model_harmonization=True,
            privacy_preservation="differential",
            max_rounds=8,
            convergence_criteria={"global_accuracy": 0.92},
        )

        logger.info(f"Created session: {self.session.name} ({self.session.session_id})")
        return self.session

    async def simulate_federated_training_round(
        self, round_number: int
    ) -> Dict[str, Any]:
        """Simulate a complete federated training round."""
        logger.info(f"Starting federated training round {round_number}")

        # Start the round
        round_state = await self.coordinator.start_federated_round(
            self.session.session_id
        )

        # Simulate participant training and model updates
        await self._simulate_participant_training(round_state)

        # Wait for aggregation to complete
        await self._wait_for_round_completion(round_state)

        # Get round results
        round_results = self.coordinator.get_session_status(self.session.session_id)

        logger.info(f"Completed federated training round {round_number}")
        return round_results

    async def _simulate_participant_training(self, round_state):
        """Simulate local training at each participant."""
        logger.info("Simulating participant training...")

        # Generate mock model weights for each participant
        for participant_id, participant in self.participants.items():
            # Simulate different model architectures based on data type
            silo = self.silos[participant.silo_id]

            if silo.data_type == "medical_imaging":
                # CNN-like architecture
                layer_weights = [
                    np.random.randn(64, 3, 7, 7),  # Conv1
                    np.random.randn(64),  # Bias1
                    np.random.randn(128, 64, 3, 3),  # Conv2
                    np.random.randn(128),  # Bias2
                    np.random.randn(256, 128, 3, 3),  # Conv3
                    np.random.randn(256),  # Bias3
                    np.random.randn(512, 256 * 4 * 4),  # FC1
                    np.random.randn(512),  # Bias4
                    np.random.randn(2, 512),  # Output
                    np.random.randn(2),  # Output bias
                ]
            elif silo.data_type == "genomic":
                # Deep neural network for genomic data
                layer_weights = [
                    np.random.randn(1000, 2048),  # Input layer
                    np.random.randn(1000),
                    np.random.randn(512, 1000),  # Hidden1
                    np.random.randn(512),
                    np.random.randn(256, 512),  # Hidden2
                    np.random.randn(256),
                    np.random.randn(128, 256),  # Hidden3
                    np.random.randn(128),
                    np.random.randn(2, 128),  # Output
                    np.random.randn(2),
                ]
            else:
                # Standard neural network for tabular data
                layer_weights = [
                    np.random.randn(128, 50),  # Input layer
                    np.random.randn(128),
                    np.random.randn(64, 128),  # Hidden1
                    np.random.randn(64),
                    np.random.randn(32, 64),  # Hidden2
                    np.random.randn(32),
                    np.random.randn(2, 32),  # Output
                    np.random.randn(2),
                ]

            # Add some noise to simulate training variations
            noise_factor = 0.1
            layer_weights = [
                w + np.random.randn(*w.shape) * noise_factor for w in layer_weights
            ]

            # Create model weights
            model_weights = ModelWeights(
                model_type="pytorch",
                layer_weights=layer_weights,
                layer_names=[f"layer_{i}" for i in range(len(layer_weights))],
                model_architecture={
                    "type": "feedforward",
                    "layers": len(layer_weights) // 2,
                    "data_type": silo.data_type,
                },
                metadata={
                    "participant_id": participant_id,
                    "silo_id": participant.silo_id,
                    "org_id": participant.org_id,
                    "training_samples": participant.data_characteristics.get(
                        "sample_count", 1000
                    ),
                    "training_accuracy": participant.performance_metrics.get(
                        "accuracy", 0.85
                    ),
                },
            )

            # Submit the update with mock API key
            await self.coordinator.submit_participant_update(
                session_id=self.session.session_id,
                participant_id=participant_id,
                model_weights=model_weights,
                api_key="demo_api_key_healthcare_federation",
            )

            logger.info(f"Submitted update from {participant.name}")

    async def _wait_for_round_completion(self, round_state, timeout: float = 600.0):
        """Wait for the round to complete with timeout."""
        start_time = asyncio.get_event_loop().time()

        while round_state.status != "completed":
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Round {round_state.round_number} timed out")

            await asyncio.sleep(5)  # Check every 5 seconds

            # Update round state
            current_state = self.coordinator.round_states.get(self.session.session_id)
            if current_state:
                round_state.status = current_state.status

        logger.info(f"Round {round_state.round_number} completed successfully")

    async def run_complete_demo(self):
        """Run the complete cross-silo federated learning demo."""
        logger.info("Starting Cross-Silo Federated Learning Demo")

        try:
            # 1. Setup healthcare federation
            await self.setup_healthcare_federation()

            # 2. Create federated session
            await self.create_federated_session()

            # 3. Run multiple training rounds
            for round_num in range(1, 4):  # Run 3 rounds for demo
                round_results = await self.simulate_federated_training_round(round_num)

                # Log round summary
                current_round = round_results.get("current_round", {})
                if current_round:
                    logger.info(f"Round {round_num} Summary:")
                    logger.info(
                        f"  - Silo aggregations: {len(current_round.get('silo_results', {}))}"
                    )
                    logger.info(
                        f"  - Organization aggregations: {len(current_round.get('org_results', {}))}"
                    )
                    logger.info(
                        f"  - Global aggregation: {'Completed' if current_round.get('global_result') else 'Pending'}"
                    )

            # 4. Show final statistics
            final_stats = self.coordinator.get_cross_org_statistics()
            logger.info("Final Federation Statistics:")
            logger.info(
                f"  - Total Organizations: {final_stats['organizations']['total_organizations']}"
            )
            logger.info(
                f"  - Total Silos: {final_stats['organizations']['total_silos']}"
            )
            logger.info(
                f"  - Total Participants: {final_stats['organizations']['total_participants']}"
            )
            logger.info(
                f"  - Total Sessions: {final_stats['sessions']['total_sessions']}"
            )
            logger.info(
                f"  - Completed Sessions: {final_stats['sessions']['completed_sessions']}"
            )

            logger.info("Cross-Silo Federated Learning Demo completed successfully!")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise


async def main():
    """Main function to run the demo."""
    demo = CrossSiloFederatedLearningDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
