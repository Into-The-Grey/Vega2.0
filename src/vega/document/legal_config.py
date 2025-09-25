"""
LegalConfig and ContractConfig for Vega 2.0 Legal Document Analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .legal import ClauseType


@dataclass
class LegalConfig:
    allowed_clause_types: List[ClauseType] = field(
        default_factory=lambda: list(ClauseType)
    )
    min_confidence: float = 0.7
    enable_entity_recognition: bool = True
    enable_compliance_check: bool = True
    enable_risk_assessment: bool = True
    jurisdiction: Optional[str] = None


@dataclass
class ContractConfig:
    require_effective_date: bool = True
    require_termination_date: bool = False
    require_signatures: bool = True
    allowed_jurisdictions: Optional[List[str]] = None
    min_clause_length: int = 30
    max_clause_length: int = 2000
