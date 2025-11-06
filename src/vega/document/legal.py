"""
Vega 2.0 Legal Document Intelligence Module

This module provides advanced legal document processing and analysis capabilities including:
- Contract analysis and clause extraction
- Legal document classification and risk assessment
- Compliance checking and regulatory analysis
- Integration with legal AI models and knowledge bases

Dependencies:
- transformers: For legal language models (e.g., legal-bert)
- spacy: For named entity recognition and legal entity extraction
- pandas: For structured legal data processing
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

from .base import (
    BaseDocumentProcessor,
    ProcessingContext,
    ProcessingResult,
    ConfigurableComponent,
    handle_import_error,
    MetricsCollector,
    DocumentIntelligenceError,
    ProcessingError,
    ValidationError,
)

# Skip heavy imports in test mode
import os

if os.environ.get("VEGA_TEST_MODE") == "1":
    HAS_TRANSFORMERS = False
    HAS_SPACY = False
    pipeline = AutoTokenizer = AutoModelForSequenceClassification = spacy = None
else:
    # Optional dependencies with graceful fallback
    HAS_TRANSFORMERS = handle_import_error("transformers", optional=True)
    HAS_SPACY = handle_import_error("spacy", optional=True)

    try:
        if HAS_TRANSFORMERS:
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        if HAS_SPACY:
            import spacy
    except ImportError:
        pipeline = AutoTokenizer = AutoModelForSequenceClassification = spacy = None

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Legal document types supported by the system"""

    CONTRACT = "contract"
    AGREEMENT = "agreement"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    LEASE = "lease"
    EMPLOYMENT = "employment"
    NDA = "nda"
    COMPLIANCE = "compliance"
    REGULATION = "regulation"
    OTHER = "other"


class RiskLevel(Enum):
    """Risk assessment levels for legal content"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClauseType(Enum):
    """Types of legal clauses that can be identified"""

    LIABILITY = "liability"
    TERMINATION = "termination"
    PAYMENT = "payment"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"
    DATA_PROTECTION = "data_protection"
    FORCE_MAJEURE = "force_majeure"
    INDEMNIFICATION = "indemnification"


@dataclass
class LegalConfig(ConfigurableComponent):
    """Configuration for legal document processing"""

    enable_clause_extraction: bool = True
    enable_risk_assessment: bool = True
    enable_compliance_check: bool = True
    enable_entity_extraction: bool = True
    min_confidence: float = 0.7
    supported_jurisdictions: List[str] = field(
        default_factory=lambda: ["US", "EU", "UK", "CA"]
    )
    risk_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "high": ["unlimited liability", "personal guarantee", "liquidated damages"],
            "medium": ["indemnification", "termination for convenience", "non-compete"],
            "low": ["standard warranty", "ordinary course of business"],
        }
    )
    clause_patterns: Dict[ClauseType, List[str]] = field(default_factory=dict)
    max_document_length: int = 50000
    timeout_seconds: float = 120.0
    use_transformers: bool = True
    legal_model_name: str = "nlpaueb/legal-bert-base-uncased"

    def __post_init__(self):
        if not self.clause_patterns:
            self.clause_patterns = self._default_clause_patterns()

    def _default_clause_patterns(self) -> Dict[ClauseType, List[str]]:
        """Default patterns for identifying legal clauses"""
        return {
            ClauseType.LIABILITY: [
                r"limitation of liability",
                r"liability.*limited",
                r"damages.*excluded",
                r"liability.*cap",
            ],
            ClauseType.TERMINATION: [
                r"termination",
                r"terminate.*agreement",
                r"end.*contract",
                r"expir[ey].*agreement",
            ],
            ClauseType.PAYMENT: [
                r"payment.*terms",
                r"invoice",
                r"due.*payment",
                r"fees.*payable",
            ],
            ClauseType.CONFIDENTIALITY: [
                r"confidential",
                r"non-disclosure",
                r"proprietary.*information",
                r"trade.*secret",
            ],
        }

    def validate_config(self) -> List[str]:
        errors = []
        if self.min_confidence < 0 or self.min_confidence > 1:
            errors.append("min_confidence must be between 0 and 1")
        if self.max_document_length <= 0:
            errors.append("max_document_length must be positive")
        if self.use_transformers and not HAS_TRANSFORMERS:
            errors.append("transformers library required but not available")
        return errors


@dataclass
class ComplianceConfig(ConfigurableComponent):
    """Configuration for compliance checking"""

    regulations: List[str] = field(
        default_factory=lambda: ["GDPR", "CCPA", "SOX", "HIPAA"]
    )
    compliance_rules: Dict[str, List[str]] = field(default_factory=dict)
    strict_mode: bool = False

    def validate_config(self) -> List[str]:
        errors = []
        if not self.regulations:
            errors.append("At least one regulation must be specified")
        return errors


class ContractAnalyzer(BaseDocumentProcessor[LegalConfig]):
    """
    Analyzes legal contracts for clauses, risks, and compliance issues
    """

    async def _async_initialize(self) -> None:
        """Initialize contract analysis models"""
        self.metrics = MetricsCollector()
        self.nlp = None
        self.legal_model = None

        # Initialize spaCy for entity extraction
        if HAS_SPACY and self.config.enable_entity_extraction:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except (OSError, ImportError) as e:
                self.logger.warning(f"spaCy model not available: {e}")

        # Initialize transformers model for legal analysis
        if self.config.use_transformers and HAS_TRANSFORMERS:
            try:
                self.legal_model = pipeline(
                    "text-classification",
                    model=self.config.legal_model_name,
                    tokenizer=self.config.legal_model_name,
                )
                self.logger.info("Legal BERT model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Legal model not available: {e}")

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze legal contract"""
        if not isinstance(input_data, str):
            raise ValidationError(
                "Input must be a string containing legal document text"
            )

        document_text = input_data.strip()
        if not document_text:
            raise ValidationError("Document text cannot be empty")

        if len(document_text) > self.config.max_document_length:
            raise ValidationError(
                f"Document too long (max {self.config.max_document_length} characters)"
            )

        self.metrics.start_timer("contract_analysis")

        try:
            # Document classification
            doc_type = await self._classify_document(document_text)

            # Extract clauses
            clauses = await self._extract_clauses(document_text)

            # Risk assessment
            risk_assessment = await self._assess_risks(document_text, clauses)

            # Entity extraction
            entities = await self._extract_entities(document_text)

            # Key terms identification
            key_terms = await self._identify_key_terms(document_text)

            self.metrics.end_timer("contract_analysis")

            return {
                "document_type": doc_type,
                "clauses": clauses,
                "risk_assessment": risk_assessment,
                "entities": entities,
                "key_terms": key_terms,
                "analysis_metadata": {
                    "word_count": len(document_text.split()),
                    "character_count": len(document_text),
                    "processing_time": self.metrics.get_metrics()
                    .get("contract_analysis", {})
                    .get("duration"),
                },
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("contract_analysis")
            raise ProcessingError(f"Contract analysis failed: {e}")

    async def _classify_document(self, text: str) -> DocumentType:
        """Classify the type of legal document"""
        text_lower = text.lower()

        # Simple rule-based classification
        if any(
            keyword in text_lower
            for keyword in ["agreement", "contract", "hereby agree"]
        ):
            if "employment" in text_lower or "employee" in text_lower:
                return DocumentType.EMPLOYMENT
            elif "lease" in text_lower or "rental" in text_lower:
                return DocumentType.LEASE
            elif "non-disclosure" in text_lower or "confidential" in text_lower:
                return DocumentType.NDA
            else:
                return DocumentType.CONTRACT
        elif "privacy policy" in text_lower:
            return DocumentType.PRIVACY_POLICY
        elif "terms of service" in text_lower or "terms of use" in text_lower:
            return DocumentType.TERMS_OF_SERVICE
        else:
            return DocumentType.OTHER

    async def _extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal clauses from document"""
        clauses = []

        for clause_type, patterns in self.config.clause_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()

                    clauses.append(
                        {
                            "type": clause_type.value,
                            "matched_text": match.group(0),
                            "context": context,
                            "position": match.start(),
                            "confidence": 0.8,  # Rule-based confidence
                        }
                    )

        return clauses

    async def _assess_risks(
        self, text: str, clauses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess legal risks in the document"""
        risk_factors = []
        overall_risk = RiskLevel.LOW

        text_lower = text.lower()

        # Check for high-risk keywords
        for risk_level, keywords in self.config.risk_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    risk_factors.append(
                        {
                            "keyword": keyword,
                            "risk_level": risk_level,
                            "context": self._extract_context(text, keyword),
                        }
                    )

                    # Update overall risk
                    if risk_level == "critical":
                        overall_risk = RiskLevel.CRITICAL
                    elif risk_level == "high" and overall_risk != RiskLevel.CRITICAL:
                        overall_risk = RiskLevel.HIGH
                    elif risk_level == "medium" and overall_risk == RiskLevel.LOW:
                        overall_risk = RiskLevel.MEDIUM

        # Analyze clause-based risks
        clause_risks = self._analyze_clause_risks(clauses)

        return {
            "overall_risk": overall_risk.value,
            "risk_factors": risk_factors,
            "clause_risks": clause_risks,
            "risk_score": self._calculate_risk_score(risk_factors, clause_risks),
        }

    def _extract_context(self, text: str, keyword: str, context_size: int = 100) -> str:
        """Extract context around a keyword"""
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return ""

        start = max(0, idx - context_size)
        end = min(len(text), idx + len(keyword) + context_size)
        return text[start:end].strip()

    def _analyze_clause_risks(
        self, clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze risks associated with specific clauses"""
        clause_risks = []

        for clause in clauses:
            risk_level = RiskLevel.LOW

            # Assess risk based on clause type
            if clause["type"] == ClauseType.LIABILITY.value:
                if "unlimited" in clause["context"].lower():
                    risk_level = RiskLevel.HIGH
                elif "limited" in clause["context"].lower():
                    risk_level = RiskLevel.LOW
                else:
                    risk_level = RiskLevel.MEDIUM

            elif clause["type"] == ClauseType.TERMINATION.value:
                if "for convenience" in clause["context"].lower():
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW

            clause_risks.append(
                {
                    "clause_type": clause["type"],
                    "risk_level": risk_level.value,
                    "reasoning": f"Risk assessment for {clause['type']} clause",
                }
            )

        return clause_risks

    def _calculate_risk_score(
        self, risk_factors: List[Dict], clause_risks: List[Dict]
    ) -> float:
        """Calculate overall risk score (0-1)"""
        score = 0.0

        # Base score from risk factors
        for factor in risk_factors:
            if factor["risk_level"] == "critical":
                score += 0.3
            elif factor["risk_level"] == "high":
                score += 0.2
            elif factor["risk_level"] == "medium":
                score += 0.1

        # Additional score from clause risks
        for risk in clause_risks:
            if risk["risk_level"] == "high":
                score += 0.15
            elif risk["risk_level"] == "medium":
                score += 0.05

        return min(score, 1.0)

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal entities from document"""
        entities = []

        if self.nlp:
            doc = self.nlp(text[:1000])  # Limit text for performance
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]:
                    entities.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "confidence": 0.9,  # spaCy confidence
                        }
                    )
        else:
            # Fallback: simple regex-based entity extraction
            patterns = {
                "DATE": r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
                "MONEY": r"\$[\d,]+\.?\d*",
                "ORGANIZATION": r"\b[A-Z][a-zA-Z\s&]+(?:Inc\.|LLC|Corp\.|Ltd\.)\b",
            }

            for entity_type, pattern in patterns.items():
                for match in re.finditer(pattern, text):
                    entities.append(
                        {
                            "text": match.group(0),
                            "label": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.7,  # Lower confidence for regex
                        }
                    )

        return entities

    async def _identify_key_terms(self, text: str) -> List[Dict[str, Any]]:
        """Identify key legal terms and definitions"""
        key_terms = []

        # Look for definition patterns
        definition_patterns = [
            r'"([^"]+)"\s+means\s+([^.]+\.)',
            r'"([^"]+)"\s+shall mean\s+([^.]+\.)',
            r"([A-Z][a-zA-Z\s]+)\s+means\s+([^.]+\.)",
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()

                key_terms.append(
                    {
                        "term": term,
                        "definition": definition,
                        "position": match.start(),
                        "type": "definition",
                    }
                )

        return key_terms


class ComplianceChecker(BaseDocumentProcessor[ComplianceConfig]):
    """
    Checks legal documents for compliance with various regulations
    """

    async def _async_initialize(self) -> None:
        """Initialize compliance checking"""
        self.metrics = MetricsCollector()
        self.compliance_patterns = self._load_compliance_patterns()

    def _load_compliance_patterns(self) -> Dict[str, List[str]]:
        """Load compliance checking patterns"""
        return {
            "GDPR": [
                r"data subject rights",
                r"lawful basis",
                r"data protection officer",
                r"privacy by design",
            ],
            "CCPA": [
                r"consumer rights",
                r"personal information",
                r"opt-out",
                r"do not sell",
            ],
            "SOX": [
                r"internal controls",
                r"financial reporting",
                r"audit committee",
                r"disclosure controls",
            ],
            "HIPAA": [
                r"protected health information",
                r"business associate",
                r"minimum necessary",
                r"security rule",
            ],
        }

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Check document compliance"""
        if not isinstance(input_data, dict):
            raise ValidationError(
                "Input must be a dictionary with 'document' and 'regulations' keys"
            )

        document = input_data.get("document", "")
        regulations = input_data.get("regulations", self.config.regulations)

        if not document:
            raise ValidationError("Document text cannot be empty")

        self.metrics.start_timer("compliance_check")

        try:
            compliance_results = {}

            for regulation in regulations:
                if regulation in self.compliance_patterns:
                    result = await self._check_regulation_compliance(
                        document, regulation
                    )
                    compliance_results[regulation] = result

            # Overall compliance score
            overall_score = self._calculate_overall_compliance(compliance_results)

            self.metrics.end_timer("compliance_check")

            return {
                "compliance_results": compliance_results,
                "overall_compliance_score": overall_score,
                "recommendations": self._generate_compliance_recommendations(
                    compliance_results
                ),
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("compliance_check")
            raise ProcessingError(f"Compliance check failed: {e}")

    async def _check_regulation_compliance(
        self, document: str, regulation: str
    ) -> Dict[str, Any]:
        """Check compliance with specific regulation"""
        patterns = self.compliance_patterns.get(regulation, [])
        matches = []

        for pattern in patterns:
            if re.search(pattern, document, re.IGNORECASE):
                matches.append(pattern)

        compliance_score = len(matches) / len(patterns) if patterns else 0.0

        return {
            "regulation": regulation,
            "compliance_score": compliance_score,
            "matched_requirements": matches,
            "missing_requirements": [p for p in patterns if p not in matches],
            "status": "compliant" if compliance_score >= 0.8 else "non-compliant",
        }

    def _calculate_overall_compliance(self, results: Dict[str, Dict]) -> float:
        """Calculate overall compliance score"""
        if not results:
            return 0.0

        scores = [result["compliance_score"] for result in results.values()]
        return sum(scores) / len(scores)

    def _generate_compliance_recommendations(
        self, results: Dict[str, Dict]
    ) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        for regulation, result in results.items():
            if result["compliance_score"] < 0.8:
                recommendations.append(
                    f"Improve {regulation} compliance by addressing: {', '.join(result['missing_requirements'][:3])}"
                )

        return recommendations


class LegalDocumentAI:
    """
    Main interface for legal document intelligence capabilities
    """

    def __init__(
        self,
        legal_config: Optional[LegalConfig] = None,
        compliance_config: Optional[ComplianceConfig] = None,
    ):
        self.legal_config = legal_config or LegalConfig()
        self.compliance_config = compliance_config or ComplianceConfig()

        # Initialize processors
        self.contract_analyzer = ContractAnalyzer(self.legal_config)
        self.compliance_checker = ComplianceChecker(self.compliance_config)

        # Track initialization state
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all processors"""
        await asyncio.gather(
            self.contract_analyzer.initialize(), self.compliance_checker.initialize()
        )
        self.is_initialized = True

    async def process_document(
        self,
        document: Union[str, Dict[str, Any], ProcessingContext],
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """
        Generic document processing that routes to appropriate analyzer.

        For legal documents, defaults to contract analysis unless document metadata
        indicates compliance checking.
        """
        # Handle ProcessingContext as document
        if isinstance(document, ProcessingContext):
            text = document.metadata.get("content", "")
            ctx = document
        elif isinstance(document, dict):
            text = document.get("content", "") or document.get("text", "")
            doc_type = document.get("type", "contract").lower()
            ctx = context or ProcessingContext()
        else:
            text = str(document)
            doc_type = "contract"
            ctx = context or ProcessingContext()

        # Validate input
        if not text or not text.strip():
            return ProcessingResult(
                success=False,
                context=ctx,
                data={"error": "Empty content provided for legal document processing"},
                errors=["Empty content provided for legal document processing"],
            )

        # Route based on document type
        if (
            isinstance(document, dict)
            and "compliance" in document.get("type", "").lower()
        ):
            return await self.check_compliance(text, context=ctx)
        elif context and "compliance" in context.metadata.get("document_category", ""):
            return await self.check_compliance(text, context=ctx)
        else:
            return await self.analyze_contract(text, context=ctx)

    async def analyze_contract(
        self, document_text: str, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """Analyze a legal contract"""
        return await self.contract_analyzer.process(document_text, context)

    async def check_compliance(
        self,
        document_text: str,
        regulations: List[str] = None,
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """Check document compliance with regulations"""
        input_data = {
            "document": document_text,
            "regulations": regulations or self.compliance_config.regulations,
        }
        return await self.compliance_checker.process(input_data, context)

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all components"""
        checks = await asyncio.gather(
            self.contract_analyzer.health_check(),
            self.compliance_checker.health_check(),
        )

        all_healthy = all(c.get("status") == "healthy" for c in checks)

        return {
            "healthy": bool(self.is_initialized and all_healthy),
            "overall_status": (
                "healthy" if (self.is_initialized and all_healthy) else "degraded"
            ),
            "components": {
                "contract_analyzer": checks[0],
                "compliance_checker": checks[1],
            },
        }


logger = logging.getLogger(__name__)


class LegalAnalysisError(Exception):
    """Custom exception for legal analysis errors"""

    pass


class ClauseType(Enum):
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    PAYMENT = "payment"
    LIABILITY = "liability"
    FORCE_MAJEURE = "force_majeure"
    GOVERNING_LAW = "governing_law"
    NON_COMPETE = "non_compete"
    INDEMNITY = "indemnity"
    ASSIGNMENT = "assignment"
    NOTICE = "notice"
    OTHER = "other"


@dataclass
class Clause:
    type: ClauseType
    text: str
    start: int
    end: int
    confidence: float = 0.0
    extracted_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalEntity:
    name: str
    entity_type: str
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class ComplianceIssue:
    description: str
    severity: str
    clause_type: Optional[ClauseType] = None
    detected_at: Optional[int] = None


@dataclass
class LegalAnalysisConfig:
    use_legal_bert: bool = True
    use_spacy: bool = True
    confidence_threshold: float = 0.7
    clause_types: List[ClauseType] = field(default_factory=lambda: list(ClauseType))


class ClauseExtractor:
    """
    Extracts clauses from legal documents
    """

    def __init__(self, config: LegalAnalysisConfig):
        self.config = config
        self.clause_patterns = self._build_clause_patterns()

    def _build_clause_patterns(self) -> Dict[ClauseType, re.Pattern]:
        patterns = {
            ClauseType.CONFIDENTIALITY: re.compile(
                r"confidentiality|non[- ]disclosure", re.I
            ),
            ClauseType.TERMINATION: re.compile(r"termination|term of agreement", re.I),
            ClauseType.PAYMENT: re.compile(r"payment|compensation|fees", re.I),
            ClauseType.LIABILITY: re.compile(
                r"liability|limitation of liability", re.I
            ),
            ClauseType.FORCE_MAJEURE: re.compile(r"force majeure", re.I),
            ClauseType.GOVERNING_LAW: re.compile(r"governing law|jurisdiction", re.I),
            ClauseType.NON_COMPETE: re.compile(r"non[- ]compete|exclusivity", re.I),
            ClauseType.INDEMNITY: re.compile(r"indemnity|indemnification", re.I),
            ClauseType.ASSIGNMENT: re.compile(r"assignment", re.I),
            ClauseType.NOTICE: re.compile(r"notice", re.I),
        }
        return patterns

    def extract_clauses(self, text: str) -> List[Clause]:
        clauses = []
        for clause_type, pattern in self.clause_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                # Extract the full clause (simple heuristic: next period or newline)
                clause_text = self._extract_full_clause(text, start)
                clauses.append(
                    Clause(
                        type=clause_type,
                        text=clause_text,
                        start=start,
                        end=start + len(clause_text),
                        confidence=0.9,
                    )
                )
        return clauses

    def _extract_full_clause(self, text: str, start: int) -> str:
        # Heuristic: extract until next double newline or period
        end = text.find("\n\n", start)
        if end == -1:
            end = text.find(".\n", start)
        if end == -1:
            end = text.find(".", start)
        if end == -1:
            end = len(text)
        return text[start:end].strip()


class LegalEntityRecognizer:
    """
    Recognizes legal entities in documents
    """

    def __init__(self, config: LegalAnalysisConfig):
        self.config = config
        self.nlp = (
            spacy.load("en_core_web_sm") if HAS_SPACY and config.use_spacy else None
        )

    def recognize_entities(self, text: str) -> List[LegalEntity]:
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE"]:
                    entities.append(
                        LegalEntity(
                            name=ent.text,
                            entity_type=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.85,
                        )
                    )
        return entities


class LegacyComplianceChecker:
    """
    Checks for compliance issues in legal documents
    """

    def __init__(self, config: LegalAnalysisConfig):
        self.config = config

    def check_compliance(self, clauses: List[Clause]) -> List[ComplianceIssue]:
        issues = []
        for clause in clauses:
            if clause.type == ClauseType.CONFIDENTIALITY and len(clause.text) < 50:
                issues.append(
                    ComplianceIssue(
                        description="Confidentiality clause too short",
                        severity="medium",
                        clause_type=clause.type,
                        detected_at=clause.start,
                    )
                )
            if clause.type == ClauseType.PAYMENT and not re.search(
                r"\$|USD|EUR|amount", clause.text, re.I
            ):
                issues.append(
                    ComplianceIssue(
                        description="Payment clause missing amount",
                        severity="high",
                        clause_type=clause.type,
                        detected_at=clause.start,
                    )
                )
        return issues


class LegacyContractAnalyzer:
    """
    Analyzes contracts for obligations, risks, and key terms
    """

    def __init__(self, config: LegalAnalysisConfig):
        self.config = config

    def analyze_contract(self, text: str) -> Dict[str, Any]:
        # Extract dates
        effective_date = self._extract_date(text, r"effective date[:\s]+([\w\s,]+)")
        termination_date = self._extract_date(text, r"termination date[:\s]+([\w\s,]+)")
        # Find obligations (simple heuristic)
        obligations = re.findall(r"\bshall\b[^.]+\.", text, re.I)
        # Find risks (look for 'risk', 'liability', 'penalty')
        risks = re.findall(r"risk|liability|penalty", text, re.I)
        return {
            "effective_date": effective_date,
            "termination_date": termination_date,
            "obligations": obligations,
            "risks": risks,
        }

    def _extract_date(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.I)
        if match:
            date_str = match.group(1)
            try:
                return str(datetime.datetime.strptime(date_str.strip(), "%B %d, %Y"))
            except Exception:
                return date_str.strip()
        return None


class LegalDocumentAnalyzer:
    """
    Main legal document analysis system
    """

    def __init__(self, config: Optional[LegalAnalysisConfig] = None):
        self.config = config or LegalAnalysisConfig()
        self.clause_extractor = ClauseExtractor(self.config)
        self.entity_recognizer = LegalEntityRecognizer(self.config)
        self.compliance_checker = ComplianceChecker(self.config)
        self.contract_analyzer = ContractAnalyzer(self.config)

    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            # Extract clauses
            clauses = self.clause_extractor.extract_clauses(text)
            # Recognize entities
            entities = self.entity_recognizer.recognize_entities(text)
            # Check compliance
            compliance_issues = self.compliance_checker.check_compliance(clauses)
            # Analyze contract
            contract_info = self.contract_analyzer.analyze_contract(text)
            return {
                "clauses": [clause.__dict__ for clause in clauses],
                "entities": [entity.__dict__ for entity in entities],
                "compliance_issues": [issue.__dict__ for issue in compliance_issues],
                "contract_info": contract_info,
            }
        except Exception as e:
            logger.error(f"Legal analysis error: {e}")
            return {"error": str(e)}


# Example usage and testing
def demo():
    print("Legal Document Analysis Demo\n" + "=" * 40)
    analyzer = LegalDocumentAnalyzer()
    sample_text = """
    This Agreement is made effective as of January 1, 2024. The parties agree to the following terms.\n\n
    1. Confidentiality: Each party shall keep all information confidential.\n\n
    2. Payment: The client shall pay $10,000 within 30 days.\n\n
    3. Termination: This agreement may be terminated by either party with 30 days notice.\n\n
    4. Governing Law: This agreement is governed by the laws of California.\n\n
    """
    result = analyzer.analyze(sample_text)
    print("Clauses:")
    for clause in result["clauses"]:
        print(f"  - {clause['type']}: {clause['text'][:60]}...")
    print("Entities:")
    for entity in result["entities"]:
        print(f"  - {entity['name']} ({entity['entity_type']})")
    print("Compliance Issues:")
    for issue in result["compliance_issues"]:
        print(f"  - {issue['description']} (Severity: {issue['severity']})")
    print("Contract Info:")
    for k, v in result["contract_info"].items():
        print(f"  - {k}: {v}")
    print("\nLegal Document Analysis demo completed successfully!")


if __name__ == "__main__":
    demo()
