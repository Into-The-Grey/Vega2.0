"""
Cross-Silo Federated Learning Coordinator

Orchestrates hierarchical federated learning across organizations, silos, and participants.
Manages the complete workflow from participant registration to global model aggregation.

Design Principles:
- Three-tier coordination: Global → Organization → Silo
- Asynchronous coordination with configurable timeouts
- Enterprise-grade security and audit capabilities
- Flexible aggregation strategies per level
- Cross-domain adaptation support
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .cross_silo import (
    Organization,
    Silo,
    HierarchicalParticipant,
    CrossSiloSession,
    FederationLevel,
    OrganizationRole,
    OrganizationManager,
)
from .hierarchical_aggregation import (
    HierarchicalAggregator,
    AggregationResult,
    LevelAggregationConfig,
)
from .model_serialization import ModelWeights, ModelSerializer
from .communication import CommunicationManager, FederatedMessage
from .coordinator import SessionStatus, AggregationStrategy
from .security import (
    audit_log,
    check_api_key,
    is_anomalous_update,
    check_model_consistency,
    validate_model_update_pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossSiloCoordinationConfig:
    """Configuration for cross-silo coordination."""

    # Timing configuration
    silo_aggregation_timeout: float = 300.0  # 5 minutes
    org_aggregation_timeout: float = 600.0  # 10 minutes
    global_aggregation_timeout: float = 900.0  # 15 minutes

    # Participation requirements
    min_organizations: int = 2
    min_silos_per_org: int = 1
    min_participants_per_silo: int = 2

    # Round configuration
    max_rounds: int = 50
    convergence_threshold: float = 0.001
    patience_rounds: int = 5

    # Security and validation
    require_all_levels: bool = True  # Require aggregation at all levels
    byzantine_tolerance_global: float = 0.3
    model_validation_strict: bool = True

    # Cross-domain settings
    enable_domain_adaptation: bool = True
    harmonization_rounds: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "silo_aggregation_timeout": self.silo_aggregation_timeout,
            "org_aggregation_timeout": self.org_aggregation_timeout,
            "global_aggregation_timeout": self.global_aggregation_timeout,
            "min_organizations": self.min_organizations,
            "min_silos_per_org": self.min_silos_per_org,
            "min_participants_per_silo": self.min_participants_per_silo,
            "max_rounds": self.max_rounds,
            "convergence_threshold": self.convergence_threshold,
            "patience_rounds": self.patience_rounds,
            "require_all_levels": self.require_all_levels,
            "byzantine_tolerance_global": self.byzantine_tolerance_global,
            "model_validation_strict": self.model_validation_strict,
            "enable_domain_adaptation": self.enable_domain_adaptation,
            "harmonization_rounds": self.harmonization_rounds,
        }


@dataclass
class CrossSiloRoundState:
    """State tracking for a cross-silo federation round."""

    session_id: str
    round_number: int
    status: str = (
        "initializing"  # "initializing", "silo_aggregation", "org_aggregation", "global_aggregation", "completed", "failed"
    )

    # Participant updates
    participant_updates: Dict[str, Dict[str, ModelWeights]] = field(
        default_factory=dict
    )  # silo_id -> {participant_id -> weights}

    # Aggregation results
    silo_results: Dict[str, AggregationResult] = field(
        default_factory=dict
    )  # silo_id -> result
    org_results: Dict[str, AggregationResult] = field(
        default_factory=dict
    )  # org_id -> result
    global_result: Optional[AggregationResult] = None

    # Timing
    round_start_time: float = field(default_factory=time.time)
    phase_start_times: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    participation_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "round_number": self.round_number,
            "status": self.status,
            "participant_updates": {
                silo_id: {pid: weights.to_dict() for pid, weights in updates.items()}
                for silo_id, updates in self.participant_updates.items()
            },
            "silo_results": {
                sid: result.to_dict() for sid, result in self.silo_results.items()
            },
            "org_results": {
                oid: result.to_dict() for oid, result in self.org_results.items()
            },
            "global_result": (
                self.global_result.to_dict() if self.global_result else None
            ),
            "round_start_time": self.round_start_time,
            "phase_start_times": self.phase_start_times,
            "convergence_metrics": self.convergence_metrics,
            "participation_stats": self.participation_stats,
        }


class CrossSiloCoordinator:
    """Coordinates cross-silo federated learning across the hierarchy."""

    def __init__(self, config: CrossSiloCoordinationConfig = None):
        """Initialize cross-silo coordinator."""
        self.config = config or CrossSiloCoordinationConfig()
        self.org_manager = OrganizationManager()
        self.hierarchical_aggregator = HierarchicalAggregator(self.org_manager)
        self.communication_manager = CommunicationManager()

        # Session management
        self.active_sessions: Dict[str, CrossSiloSession] = {}
        self.round_states: Dict[str, CrossSiloRoundState] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "session_started": [],
            "round_completed": [],
            "session_completed": [],
            "aggregation_completed": [],
            "participant_joined": [],
            "participant_left": [],
        }

        self.logger = logging.getLogger(__name__ + ".CrossSiloCoordinator")

    async def create_cross_silo_session(
        self,
        name: str,
        description: str,
        participating_orgs: List[str],
        participating_silos: Dict[str, List[str]],
        **kwargs,
    ) -> CrossSiloSession:
        """Create a new cross-silo federated learning session."""
        session_id = f"cross_silo_{uuid.uuid4().hex[:12]}"

        # Validate participating organizations and silos
        for org_id in participating_orgs:
            if org_id not in self.org_manager.organizations:
                raise ValueError(f"Organization {org_id} not found")

        for org_id, silo_ids in participating_silos.items():
            if org_id not in participating_orgs:
                raise ValueError(
                    f"Organization {org_id} not in participating organizations"
                )

            for silo_id in silo_ids:
                if silo_id not in self.org_manager.silos:
                    raise ValueError(f"Silo {silo_id} not found")
                if self.org_manager.silos[silo_id].org_id != org_id:
                    raise ValueError(
                        f"Silo {silo_id} does not belong to organization {org_id}"
                    )

        # Validate minimum participation requirements
        if len(participating_orgs) < self.config.min_organizations:
            raise ValueError(
                f"Insufficient organizations: {len(participating_orgs)} < {self.config.min_organizations}"
            )

        for org_id, silo_ids in participating_silos.items():
            if len(silo_ids) < self.config.min_silos_per_org:
                raise ValueError(
                    f"Insufficient silos for organization {org_id}: {len(silo_ids)} < {self.config.min_silos_per_org}"
                )

        # Create session
        session = CrossSiloSession(
            session_id=session_id,
            name=name,
            description=description,
            participating_orgs=participating_orgs,
            participating_silos=participating_silos,
            **kwargs,
        )

        # Setup aggregation levels
        session.aggregation_levels = {
            FederationLevel.SILO: AggregationStrategy.FEDERATED_AVERAGING,
            FederationLevel.ORGANIZATION: AggregationStrategy.FEDERATED_AVERAGING,
            FederationLevel.GLOBAL: AggregationStrategy.FEDERATED_AVERAGING,
        }

        self.active_sessions[session_id] = session

        await audit_log(
            "cross_silo_session_created",
            {
                "session_id": session_id,
                "name": name,
                "participating_orgs": participating_orgs,
                "participating_silos": participating_silos,
            },
            "system",
        )

        # Notify event handlers
        await self._trigger_event("session_started", session)

        self.logger.info(f"Created cross-silo session: {name} ({session_id})")
        return session

    async def start_federated_round(
        self, session_id: str, initial_model: Optional[ModelWeights] = None
    ) -> CrossSiloRoundState:
        """Start a new round of cross-silo federated learning."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        round_number = session.current_round + 1

        # Create round state
        round_state = CrossSiloRoundState(
            session_id=session_id, round_number=round_number
        )
        round_state.phase_start_times["initialization"] = time.time()

        self.round_states[session_id] = round_state

        await audit_log(
            "federated_round_started",
            {
                "session_id": session_id,
                "round_number": round_number,
                "initial_model_provided": initial_model is not None,
            },
            "system",
        )

        # Broadcast initial model to all participants
        if initial_model:
            await self._broadcast_initial_model(session, initial_model, round_number)

        # Update session
        session.current_round = round_number
        session.updated_at = time.time()
        session.status = SessionStatus.RUNNING

        self.logger.info(f"Started round {round_number} for session {session_id}")
        return round_state

    async def submit_participant_update(
        self,
        session_id: str,
        participant_id: str,
        model_weights: ModelWeights,
        api_key: str,
    ) -> bool:
        """Submit a participant's model update."""
        # Validate API key
        if not await check_api_key(api_key):
            raise ValueError("Invalid API key")

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        if session_id not in self.round_states:
            raise ValueError(f"No active round for session {session_id}")

        # Find participant and their silo
        participant = self.org_manager.participants.get(participant_id)
        if not participant:
            raise ValueError(f"Participant {participant_id} not found")

        if not participant.silo_id:
            raise ValueError(f"Participant {participant_id} not assigned to a silo")

        # Validate participant is part of this session
        session = self.active_sessions[session_id]
        org_silos = session.participating_silos.get(participant.org_id, [])
        if participant.silo_id not in org_silos:
            raise ValueError(
                f"Participant {participant_id} not authorized for this session"
            )

        # Security validation
        validation_result = await validate_model_update_pipeline(model_weights, {})
        if not validation_result["valid"]:
            raise ValueError(f"Model validation failed: {validation_result['reason']}")

        # Store the update
        round_state = self.round_states[session_id]
        silo_id = participant.silo_id

        if silo_id not in round_state.participant_updates:
            round_state.participant_updates[silo_id] = {}

        round_state.participant_updates[silo_id][participant_id] = model_weights

        await audit_log(
            "participant_update_submitted",
            {
                "session_id": session_id,
                "participant_id": participant_id,
                "silo_id": silo_id,
                "org_id": participant.org_id,
                "round_number": round_state.round_number,
            },
            participant_id,
        )

        self.logger.info(
            f"Received update from participant {participant_id} in silo {silo_id}"
        )

        # Check if we can proceed with aggregation
        await self._check_aggregation_readiness(session_id)

        return True

    async def _check_aggregation_readiness(self, session_id: str):
        """Check if we have enough updates to proceed with aggregation."""
        session = self.active_sessions[session_id]
        round_state = self.round_states[session_id]

        # Check each silo for readiness
        ready_silos = []
        for org_id in session.participating_orgs:
            org_silos = session.participating_silos.get(org_id, [])
            for silo_id in org_silos:
                participant_count = len(
                    round_state.participant_updates.get(silo_id, {})
                )
                if participant_count >= self.config.min_participants_per_silo:
                    ready_silos.append(silo_id)

        # Start silo-level aggregation for ready silos
        if ready_silos and round_state.status == "initializing":
            round_state.status = "silo_aggregation"
            round_state.phase_start_times["silo_aggregation"] = time.time()

            # Start aggregation tasks
            await asyncio.gather(
                *[self._aggregate_silo(session_id, silo_id) for silo_id in ready_silos]
            )

    async def _aggregate_silo(self, session_id: str, silo_id: str):
        """Aggregate participant updates within a silo."""
        try:
            round_state = self.round_states[session_id]
            participant_updates = round_state.participant_updates.get(silo_id, {})

            if len(participant_updates) < self.config.min_participants_per_silo:
                self.logger.warning(f"Insufficient participants in silo {silo_id}")
                return

            # Perform silo-level aggregation
            result = await self.hierarchical_aggregator.aggregate_silo_level(
                session_id, silo_id, participant_updates, round_state.round_number
            )

            round_state.silo_results[silo_id] = result

            self.logger.info(f"Completed silo aggregation for {silo_id}")

            # Check if we can proceed to organization-level aggregation
            await self._check_organization_aggregation_readiness(session_id)

        except Exception as e:
            self.logger.error(f"Error in silo aggregation for {silo_id}: {e}")
            await audit_log(
                "silo_aggregation_failed",
                {"session_id": session_id, "silo_id": silo_id, "error": str(e)},
                "system",
            )

    async def _check_organization_aggregation_readiness(self, session_id: str):
        """Check if organizations are ready for aggregation."""
        session = self.active_sessions[session_id]
        round_state = self.round_states[session_id]

        ready_orgs = []
        for org_id in session.participating_orgs:
            org_silos = session.participating_silos.get(org_id, [])
            completed_silos = [
                silo_id for silo_id in org_silos if silo_id in round_state.silo_results
            ]

            if len(completed_silos) >= self.config.min_silos_per_org:
                ready_orgs.append(org_id)

        # Start organization-level aggregation
        if ready_orgs and round_state.status == "silo_aggregation":
            round_state.status = "org_aggregation"
            round_state.phase_start_times["org_aggregation"] = time.time()

            await asyncio.gather(
                *[
                    self._aggregate_organization(session_id, org_id)
                    for org_id in ready_orgs
                ]
            )

    async def _aggregate_organization(self, session_id: str, org_id: str):
        """Aggregate silo updates within an organization."""
        try:
            session = self.active_sessions[session_id]
            round_state = self.round_states[session_id]

            # Collect silo updates for this organization
            org_silos = session.participating_silos.get(org_id, [])
            silo_updates = {}

            for silo_id in org_silos:
                if silo_id in round_state.silo_results:
                    silo_updates[silo_id] = round_state.silo_results[
                        silo_id
                    ].aggregated_weights

            if len(silo_updates) < self.config.min_silos_per_org:
                self.logger.warning(f"Insufficient silos for organization {org_id}")
                return

            # Perform organization-level aggregation
            result = await self.hierarchical_aggregator.aggregate_organization_level(
                session_id, org_id, silo_updates, round_state.round_number
            )

            round_state.org_results[org_id] = result

            self.logger.info(f"Completed organization aggregation for {org_id}")

            # Check if we can proceed to global aggregation
            await self._check_global_aggregation_readiness(session_id)

        except Exception as e:
            self.logger.error(f"Error in organization aggregation for {org_id}: {e}")
            await audit_log(
                "organization_aggregation_failed",
                {"session_id": session_id, "org_id": org_id, "error": str(e)},
                "system",
            )

    async def _check_global_aggregation_readiness(self, session_id: str):
        """Check if we're ready for global aggregation."""
        session = self.active_sessions[session_id]
        round_state = self.round_states[session_id]

        completed_orgs = len(round_state.org_results)

        if (
            completed_orgs >= self.config.min_organizations
            and round_state.status == "org_aggregation"
        ):

            round_state.status = "global_aggregation"
            round_state.phase_start_times["global_aggregation"] = time.time()

            await self._aggregate_global(session_id)

    async def _aggregate_global(self, session_id: str):
        """Perform global aggregation across organizations."""
        try:
            round_state = self.round_states[session_id]

            # Collect organization updates
            org_updates = {
                org_id: result.aggregated_weights
                for org_id, result in round_state.org_results.items()
            }

            if len(org_updates) < self.config.min_organizations:
                self.logger.warning(
                    f"Insufficient organizations for global aggregation"
                )
                return

            # Perform global aggregation
            result = await self.hierarchical_aggregator.aggregate_global_level(
                session_id, org_updates, round_state.round_number
            )

            round_state.global_result = result
            round_state.status = "completed"

            # Update session
            session = self.active_sessions[session_id]
            session.performance_history.append(
                {
                    "round": round_state.round_number,
                    "global_result": result.to_dict(),
                    "timestamp": time.time(),
                }
            )

            self.logger.info(f"Completed global aggregation for session {session_id}")

            # Trigger event handlers
            await self._trigger_event("round_completed", round_state)
            await self._trigger_event("aggregation_completed", result)

            # Check for convergence
            await self._check_convergence(session_id)

        except Exception as e:
            self.logger.error(
                f"Error in global aggregation for session {session_id}: {e}"
            )
            round_state.status = "failed"
            await audit_log(
                "global_aggregation_failed",
                {"session_id": session_id, "error": str(e)},
                "system",
            )

    async def _check_convergence(self, session_id: str):
        """Check if the session has converged."""
        session = self.active_sessions[session_id]
        round_state = self.round_states[session_id]

        if round_state.global_result:
            convergence_rate = round_state.global_result.performance_metrics.get(
                "convergence_rate", 0.0
            )

            if convergence_rate < self.config.convergence_threshold:
                session.status = SessionStatus.COMPLETED
                await self._trigger_event("session_completed", session)
                self.logger.info(
                    f"Session {session_id} converged after {session.current_round} rounds"
                )

            elif session.current_round >= self.config.max_rounds:
                session.status = SessionStatus.COMPLETED
                await self._trigger_event("session_completed", session)
                self.logger.info(f"Session {session_id} completed maximum rounds")

    async def _broadcast_initial_model(
        self, session: CrossSiloSession, model: ModelWeights, round_number: int
    ):
        """Broadcast initial model to all participants."""
        message = FederatedMessage(
            message_type="initial_model",
            sender_id="coordinator",
            session_id=session.session_id,
            round_number=round_number,
            payload={"model_weights": model.to_dict()},
            timestamp=time.time(),
        )

        # Broadcast to all participants
        for org_id in session.participating_orgs:
            org_silos = session.participating_silos.get(org_id, [])
            for silo_id in org_silos:
                participant_ids = self.org_manager.silo_participants.get(silo_id, set())
                for participant_id in participant_ids:
                    await self.communication_manager.send_message(
                        participant_id, message
                    )

    async def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers."""
        handlers = self.event_handlers.get(event_type, [])
        if handlers:
            await asyncio.gather(*[handler(data) for handler in handlers])

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get detailed status of a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        round_state = self.round_states.get(session_id)

        status = {
            "session": session.to_dict(),
            "current_round": round_state.to_dict() if round_state else None,
            "aggregation_statistics": self.hierarchical_aggregator.get_aggregation_statistics(),
            "organization_hierarchy": {
                org_id: self.org_manager.get_organization_hierarchy(org_id)
                for org_id in session.participating_orgs
            },
        }

        return status

    def get_cross_org_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cross-organizational statistics."""
        base_stats = self.org_manager.get_cross_org_statistics()
        aggregation_stats = self.hierarchical_aggregator.get_aggregation_statistics()

        session_stats = {
            "total_sessions": len(self.active_sessions),
            "active_sessions": len(
                [
                    s
                    for s in self.active_sessions.values()
                    if s.status == SessionStatus.RUNNING
                ]
            ),
            "completed_sessions": len(
                [
                    s
                    for s in self.active_sessions.values()
                    if s.status == SessionStatus.COMPLETED
                ]
            ),
            "sessions_by_domain": {},
        }

        # Analyze sessions by participating domains
        for session in self.active_sessions.values():
            domains = set()
            for org_id in session.participating_orgs:
                org = self.org_manager.organizations.get(org_id)
                if org:
                    domains.add(org.domain)

            domain_key = "_".join(sorted(domains))
            if domain_key not in session_stats["sessions_by_domain"]:
                session_stats["sessions_by_domain"][domain_key] = 0
            session_stats["sessions_by_domain"][domain_key] += 1

        return {
            "organizations": base_stats,
            "aggregation": aggregation_stats,
            "sessions": session_stats,
        }
