"""
Cross-Silo Federated Learning - Organizational Hierarchy

Implements organization-level federation with hierarchical aggregation for enterprise
scenarios. Supports multi-level federation where organizations aggregate locally
before contributing to global aggregation.

Design Principles:
- Three-tier hierarchy: Global → Organization → Participants
- Organizations act as intermediate aggregators
- Support for cross-domain federation
- Flexible aggregation strategies at each level
- Enterprise-grade security and audit capabilities
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .model_serialization import ModelWeights, ModelSerializer
from .communication import CommunicationManager, FederatedMessage
from .coordinator import FederatedSession, SessionStatus, AggregationStrategy
from .fedavg import FedAvg, FedAvgConfig
from .security import (
    audit_log,
    check_api_key,
    is_anomalous_update,
    check_model_consistency,
    validate_model_update_pipeline,
    verify_model_signature,
    create_model_signature,
    compute_model_hash,
)

logger = logging.getLogger(__name__)


class OrganizationRole(Enum):
    """Roles within an organization."""

    ADMIN = "admin"
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class FederationLevel(Enum):
    """Levels in the hierarchical federation."""

    GLOBAL = "global"
    ORGANIZATION = "organization"
    SILO = "silo"


@dataclass
class Organization:
    """Represents an organization in the federated learning hierarchy."""

    org_id: str
    name: str
    description: str
    domain: str  # e.g., "healthcare", "finance", "retail"
    admin_contact: str
    created_at: float
    updated_at: float
    is_active: bool = True

    # Organizational policies
    min_participants: int = 2
    max_participants: int = 100
    data_sharing_policy: str = "strict"  # "strict", "moderate", "open"
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING

    # Security configuration
    requires_approval: bool = True
    encryption_required: bool = True
    audit_level: str = "full"  # "minimal", "standard", "full"

    # Organizational statistics
    total_participants: int = 0
    active_sessions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "admin_contact": self.admin_contact,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "min_participants": self.min_participants,
            "max_participants": self.max_participants,
            "data_sharing_policy": self.data_sharing_policy,
            "aggregation_strategy": self.aggregation_strategy.value,
            "requires_approval": self.requires_approval,
            "encryption_required": self.encryption_required,
            "audit_level": self.audit_level,
            "total_participants": self.total_participants,
            "active_sessions": self.active_sessions,
            "metadata": self.metadata,
        }


@dataclass
class Silo:
    """Represents a data silo within an organization."""

    silo_id: str
    org_id: str
    name: str
    description: str
    data_type: str  # e.g., "tabular", "image", "text", "time_series"
    location: str
    contact: str
    created_at: float
    updated_at: float
    is_active: bool = True

    # Silo capabilities
    supported_models: List[str] = field(default_factory=list)
    data_size: int = 0  # Number of samples
    compute_capacity: str = "medium"  # "low", "medium", "high"
    availability: str = "business_hours"  # "24/7", "business_hours", "on_demand"

    # Participant management
    participants: List[str] = field(default_factory=list)
    max_participants: int = 10

    # Security and compliance
    privacy_level: str = "high"  # "low", "medium", "high"
    compliance_standards: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "silo_id": self.silo_id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "location": self.location,
            "contact": self.contact,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "supported_models": self.supported_models,
            "data_size": self.data_size,
            "compute_capacity": self.compute_capacity,
            "availability": self.availability,
            "participants": self.participants,
            "max_participants": self.max_participants,
            "privacy_level": self.privacy_level,
            "compliance_standards": self.compliance_standards,
            "metadata": self.metadata,
        }


@dataclass
class HierarchicalParticipant:
    """Enhanced participant with organizational context."""

    participant_id: str
    org_id: str
    silo_id: Optional[str]
    name: str
    role: OrganizationRole
    contact: str
    created_at: float
    updated_at: float
    is_active: bool = True

    # Hierarchical context
    federation_level: FederationLevel = FederationLevel.SILO
    parent_coordinator: Optional[str] = None
    managed_participants: List[str] = field(default_factory=list)

    # Capabilities and resources
    supported_models: List[str] = field(default_factory=list)
    compute_resources: Dict[str, Any] = field(default_factory=dict)
    data_characteristics: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    participation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant_id": self.participant_id,
            "org_id": self.org_id,
            "silo_id": self.silo_id,
            "name": self.name,
            "role": self.role.value,
            "contact": self.contact,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "federation_level": self.federation_level.value,
            "parent_coordinator": self.parent_coordinator,
            "managed_participants": self.managed_participants,
            "supported_models": self.supported_models,
            "compute_resources": self.compute_resources,
            "data_characteristics": self.data_characteristics,
            "participation_history": self.participation_history,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
        }


@dataclass
class CrossSiloSession:
    """Extended federated session for cross-silo learning."""

    session_id: str
    name: str
    description: str

    # Hierarchical configuration
    participating_orgs: List[str]
    participating_silos: Dict[str, List[str]]  # org_id -> [silo_ids]
    federation_topology: str = "star"  # "star", "tree", "mesh"

    # Multi-level aggregation
    aggregation_levels: Dict[FederationLevel, AggregationStrategy]
    coordination_schedule: Dict[str, Any]  # Timing for each level

    # Cross-domain settings
    domain_adaptation: bool = False
    model_harmonization: bool = True
    privacy_preservation: str = (
        "differential"  # "differential", "secure_mpc", "homomorphic"
    )

    # Session management
    status: SessionStatus = SessionStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Performance tracking
    current_round: int = 0
    max_rounds: int = 20
    convergence_criteria: Dict[str, float] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "participating_orgs": self.participating_orgs,
            "participating_silos": self.participating_silos,
            "federation_topology": self.federation_topology,
            "aggregation_levels": {
                k.value: v.value for k, v in self.aggregation_levels.items()
            },
            "coordination_schedule": self.coordination_schedule,
            "domain_adaptation": self.domain_adaptation,
            "model_harmonization": self.model_harmonization,
            "privacy_preservation": self.privacy_preservation,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "convergence_criteria": self.convergence_criteria,
            "performance_history": self.performance_history,
            "metadata": self.metadata,
        }


class OrganizationManager:
    """Manages organizations, silos, and hierarchical participants."""

    def __init__(self):
        """Initialize organization manager."""
        self.organizations: Dict[str, Organization] = {}
        self.silos: Dict[str, Silo] = {}
        self.participants: Dict[str, HierarchicalParticipant] = {}

        # Organizational relationships
        self.org_silos: Dict[str, Set[str]] = {}  # org_id -> silo_ids
        self.silo_participants: Dict[str, Set[str]] = {}  # silo_id -> participant_ids

        self.logger = logging.getLogger(__name__ + ".OrganizationManager")

    async def create_organization(
        self, name: str, description: str, domain: str, admin_contact: str, **kwargs
    ) -> Organization:
        """Create a new organization."""
        org_id = f"org_{uuid.uuid4().hex[:8]}"
        current_time = time.time()

        org = Organization(
            org_id=org_id,
            name=name,
            description=description,
            domain=domain,
            admin_contact=admin_contact,
            created_at=current_time,
            updated_at=current_time,
            **kwargs,
        )

        self.organizations[org_id] = org
        self.org_silos[org_id] = set()

        await audit_log(
            "organization_created",
            {"org_id": org_id, "name": name, "domain": domain},
            "system",
        )

        self.logger.info(f"Created organization: {name} ({org_id})")
        return org

    async def create_silo(
        self,
        org_id: str,
        name: str,
        description: str,
        data_type: str,
        location: str,
        contact: str,
        **kwargs,
    ) -> Silo:
        """Create a new silo within an organization."""
        if org_id not in self.organizations:
            raise ValueError(f"Organization {org_id} not found")

        silo_id = f"silo_{uuid.uuid4().hex[:8]}"
        current_time = time.time()

        silo = Silo(
            silo_id=silo_id,
            org_id=org_id,
            name=name,
            description=description,
            data_type=data_type,
            location=location,
            contact=contact,
            created_at=current_time,
            updated_at=current_time,
            **kwargs,
        )

        self.silos[silo_id] = silo
        self.org_silos[org_id].add(silo_id)
        self.silo_participants[silo_id] = set()

        await audit_log(
            "silo_created",
            {
                "silo_id": silo_id,
                "org_id": org_id,
                "name": name,
                "data_type": data_type,
            },
            "system",
        )

        self.logger.info(f"Created silo: {name} ({silo_id}) in organization {org_id}")
        return silo

    async def register_participant(
        self,
        org_id: str,
        silo_id: Optional[str],
        name: str,
        role: OrganizationRole,
        contact: str,
        **kwargs,
    ) -> HierarchicalParticipant:
        """Register a new participant in the organization/silo."""
        if org_id not in self.organizations:
            raise ValueError(f"Organization {org_id} not found")

        if silo_id and silo_id not in self.silos:
            raise ValueError(f"Silo {silo_id} not found")

        participant_id = f"participant_{uuid.uuid4().hex[:8]}"
        current_time = time.time()

        participant = HierarchicalParticipant(
            participant_id=participant_id,
            org_id=org_id,
            silo_id=silo_id,
            name=name,
            role=role,
            contact=contact,
            created_at=current_time,
            updated_at=current_time,
            **kwargs,
        )

        self.participants[participant_id] = participant

        if silo_id:
            self.silo_participants[silo_id].add(participant_id)

        # Update organization participant count
        self.organizations[org_id].total_participants += 1
        self.organizations[org_id].updated_at = current_time

        await audit_log(
            "participant_registered",
            {
                "participant_id": participant_id,
                "org_id": org_id,
                "silo_id": silo_id,
                "name": name,
                "role": role.value,
            },
            "system",
        )

        self.logger.info(
            f"Registered participant: {name} ({participant_id}) in {org_id}/{silo_id}"
        )
        return participant

    def get_organization_hierarchy(self, org_id: str) -> Dict[str, Any]:
        """Get complete hierarchy for an organization."""
        if org_id not in self.organizations:
            return {}

        org = self.organizations[org_id]
        hierarchy = {"organization": org.to_dict(), "silos": {}, "participants": {}}

        # Add silos and their participants
        for silo_id in self.org_silos.get(org_id, set()):
            silo = self.silos[silo_id]
            hierarchy["silos"][silo_id] = silo.to_dict()

            # Add participants in this silo
            hierarchy["silos"][silo_id]["participants"] = []
            for participant_id in self.silo_participants.get(silo_id, set()):
                participant = self.participants[participant_id]
                hierarchy["silos"][silo_id]["participants"].append(
                    participant.to_dict()
                )

        # Add organization-level participants (not in specific silos)
        org_participants = [
            p
            for p in self.participants.values()
            if p.org_id == org_id and p.silo_id is None
        ]
        hierarchy["participants"] = [p.to_dict() for p in org_participants]

        return hierarchy

    def get_cross_org_statistics(self) -> Dict[str, Any]:
        """Get statistics across all organizations."""
        stats = {
            "total_organizations": len(self.organizations),
            "total_silos": len(self.silos),
            "total_participants": len(self.participants),
            "organizations_by_domain": {},
            "participants_by_role": {},
            "silos_by_data_type": {},
            "active_organizations": 0,
            "active_silos": 0,
            "active_participants": 0,
        }

        # Count by domain
        for org in self.organizations.values():
            domain = org.domain
            if domain not in stats["organizations_by_domain"]:
                stats["organizations_by_domain"][domain] = 0
            stats["organizations_by_domain"][domain] += 1

            if org.is_active:
                stats["active_organizations"] += 1

        # Count participants by role
        for participant in self.participants.values():
            role = participant.role.value
            if role not in stats["participants_by_role"]:
                stats["participants_by_role"][role] = 0
            stats["participants_by_role"][role] += 1

            if participant.is_active:
                stats["active_participants"] += 1

        # Count silos by data type
        for silo in self.silos.values():
            data_type = silo.data_type
            if data_type not in stats["silos_by_data_type"]:
                stats["silos_by_data_type"][data_type] = 0
            stats["silos_by_data_type"][data_type] += 1

            if silo.is_active:
                stats["active_silos"] += 1

        return stats
