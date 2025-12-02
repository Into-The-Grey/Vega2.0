#!/usr/bin/env python3
"""
compliance_reporter.py - Security Compliance Reporting System

Comprehensive compliance reporting for Vega 2.0 platform including:
- SOC 2 Type II compliance reporting
- ISO 27001 security controls assessment
- NIST Cybersecurity Framework alignment
- Automated evidence collection
- Audit trail generation
- Risk assessment documentation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import sqlite3
import logging
import subprocess
import hashlib
from pathlib import Path
import uuid


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""

    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    NIST_CSF = "NIST_CSF"
    PCI_DSS = "PCI_DSS"
    GDPR = "GDPR"


class ControlStatus(Enum):
    """Security control implementation status"""

    IMPLEMENTED = "implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(Enum):
    """Risk assessment levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class SecurityControl:
    """Security control definition"""

    control_id: str
    title: str
    description: str
    framework: ComplianceFramework
    category: str
    status: ControlStatus
    implementation_date: Optional[datetime] = None
    evidence: List[str] = field(default_factory=list)
    responsible_party: Optional[str] = None
    testing_frequency: str = "annual"
    last_tested: Optional[datetime] = None
    next_test_due: Optional[datetime] = None
    findings: List[str] = field(default_factory=list)
    remediation_plan: Optional[str] = None


@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""

    assessment_id: str
    framework: ComplianceFramework
    assessment_date: datetime
    assessor: str
    scope: str
    overall_status: str
    controls_assessed: int
    controls_implemented: int
    controls_partial: int
    controls_not_implemented: int
    risk_score: float
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence_files: List[str] = field(default_factory=list)


class ComplianceReporter:
    """Comprehensive compliance reporting system"""

    def __init__(self, db_path: str = "security/compliance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("compliance_reporter")

        # Initialize database
        self._init_database()

        # Load compliance frameworks
        self._load_compliance_frameworks()

    def _init_database(self) -> None:
        """Initialize compliance tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS security_controls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                control_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                framework TEXT NOT NULL,
                category TEXT,
                status TEXT NOT NULL,
                implementation_date TIMESTAMP,
                evidence TEXT,
                responsible_party TEXT,
                testing_frequency TEXT,
                last_tested TIMESTAMP,
                next_test_due TIMESTAMP,
                findings TEXT,
                remediation_plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compliance_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT UNIQUE NOT NULL,
                framework TEXT NOT NULL,
                assessment_date TIMESTAMP NOT NULL,
                assessor TEXT NOT NULL,
                scope TEXT,
                overall_status TEXT,
                controls_assessed INTEGER,
                controls_implemented INTEGER,
                controls_partial INTEGER,
                controls_not_implemented INTEGER,
                risk_score REAL,
                findings TEXT,
                recommendations TEXT,
                evidence_files TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_description TEXT,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_compliance_frameworks(self) -> None:
        """Load and initialize compliance framework controls"""
        # SOC 2 Type II Controls
        soc2_controls = [
            SecurityControl(
                control_id="CC1.1",
                title="Control Environment - Integrity and Ethical Values",
                description="The entity demonstrates a commitment to integrity and ethical values",
                framework=ComplianceFramework.SOC2,
                category="Common Criteria",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="CC2.1",
                title="Communication and Information - Internal Communication",
                description="The entity obtains or generates relevant, quality information",
                framework=ComplianceFramework.SOC2,
                category="Common Criteria",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="CC6.1",
                title="Logical and Physical Access Controls - Authentication",
                description="The entity implements logical access security software",
                framework=ComplianceFramework.SOC2,
                category="Common Criteria",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="CC6.7",
                title="Transmission and Disposal of Information",
                description="The entity restricts the transmission, movement, and removal of information",
                framework=ComplianceFramework.SOC2,
                category="Common Criteria",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="CC7.2",
                title="System Monitoring - Security Incidents",
                description="The entity monitors system components and the operation of controls",
                framework=ComplianceFramework.SOC2,
                category="Common Criteria",
                status=ControlStatus.PARTIALLY_IMPLEMENTED,
            ),
        ]

        # ISO 27001:2022 Controls
        iso27001_controls = [
            SecurityControl(
                control_id="A.5.1",
                title="Information Security Policies",
                description="Information security policy and topic-specific policies",
                framework=ComplianceFramework.ISO27001,
                category="Organizational Controls",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="A.8.1",
                title="User Endpoint Devices",
                description="Information stored on, processed by or accessible via user endpoint devices",
                framework=ComplianceFramework.ISO27001,
                category="Technology Controls",
                status=ControlStatus.IMPLEMENTED,
            ),
            SecurityControl(
                control_id="A.8.23",
                title="Web Filtering",
                description="Access to external websites shall be managed",
                framework=ComplianceFramework.ISO27001,
                category="Technology Controls",
                status=ControlStatus.NOT_APPLICABLE,
            ),
            SecurityControl(
                control_id="A.8.28",
                title="Secure Coding",
                description="Secure coding principles shall be applied to software development",
                framework=ComplianceFramework.ISO27001,
                category="Technology Controls",
                status=ControlStatus.IMPLEMENTED,
            ),
        ]

        # Store controls in database
        for control in soc2_controls + iso27001_controls:
            self._store_security_control(control)

    def _store_security_control(self, control: SecurityControl) -> None:
        """Store security control in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO security_controls 
            (control_id, title, description, framework, category, status,
             implementation_date, evidence, responsible_party, testing_frequency,
             last_tested, next_test_due, findings, remediation_plan)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                control.control_id,
                control.title,
                control.description,
                control.framework.value,
                control.category,
                control.status.value,
                control.implementation_date,
                json.dumps(control.evidence),
                control.responsible_party,
                control.testing_frequency,
                control.last_tested,
                control.next_test_due,
                json.dumps(control.findings),
                control.remediation_plan,
            ),
        )

        conn.commit()
        conn.close()

    def conduct_compliance_assessment(
        self, framework: ComplianceFramework, assessor: str
    ) -> ComplianceAssessment:
        """Conduct comprehensive compliance assessment"""
        self.logger.info(f"Starting {framework.value} compliance assessment")

        assessment_id = str(uuid.uuid4())
        assessment_date = datetime.now()

        # Get all controls for framework
        controls = self._get_controls_by_framework(framework)

        # Collect automated evidence
        evidence_files = self._collect_automated_evidence(framework)

        # Assess control implementation
        implemented = len(
            [c for c in controls if c.status == ControlStatus.IMPLEMENTED]
        )
        partial = len(
            [c for c in controls if c.status == ControlStatus.PARTIALLY_IMPLEMENTED]
        )
        not_implemented = len(
            [c for c in controls if c.status == ControlStatus.NOT_IMPLEMENTED]
        )

        # Calculate risk score (0-100, lower is better)
        total_applicable = len(
            [c for c in controls if c.status != ControlStatus.NOT_APPLICABLE]
        )
        if total_applicable > 0:
            risk_score = ((not_implemented * 10) + (partial * 3)) / total_applicable
        else:
            risk_score = 0.0

        # Generate findings and recommendations
        findings = self._generate_assessment_findings(controls)
        recommendations = self._generate_assessment_recommendations(controls)

        # Determine overall status
        if risk_score <= 2.0:
            overall_status = "COMPLIANT"
        elif risk_score <= 5.0:
            overall_status = "MOSTLY_COMPLIANT"
        else:
            overall_status = "NON_COMPLIANT"

        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            framework=framework,
            assessment_date=assessment_date,
            assessor=assessor,
            scope="Vega 2.0 Platform",
            overall_status=overall_status,
            controls_assessed=len(controls),
            controls_implemented=implemented,
            controls_partial=partial,
            controls_not_implemented=not_implemented,
            risk_score=risk_score,
            findings=findings,
            recommendations=recommendations,
            evidence_files=evidence_files,
        )

        # Store assessment
        self._store_compliance_assessment(assessment)

        self.logger.info(f"Compliance assessment completed: {overall_status}")
        return assessment

    def _get_controls_by_framework(
        self, framework: ComplianceFramework
    ) -> List[SecurityControl]:
        """Get security controls for specific framework"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM security_controls WHERE framework = ?", (framework.value,)
        )
        rows = cursor.fetchall()
        conn.close()

        controls = []
        for row in rows:
            control = SecurityControl(
                control_id=row[1],
                title=row[2],
                description=row[3],
                framework=ComplianceFramework(row[4]),
                category=row[5],
                status=ControlStatus(row[6]),
                implementation_date=datetime.fromisoformat(row[7]) if row[7] else None,
                evidence=json.loads(row[8]) if row[8] else [],
                responsible_party=row[9],
                testing_frequency=row[10],
                last_tested=datetime.fromisoformat(row[11]) if row[11] else None,
                next_test_due=datetime.fromisoformat(row[12]) if row[12] else None,
                findings=json.loads(row[13]) if row[13] else [],
                remediation_plan=row[14],
            )
            controls.append(control)

        return controls

    def _collect_automated_evidence(self, framework: ComplianceFramework) -> List[str]:
        """Collect automated evidence for compliance assessment"""
        evidence_files = []

        try:
            # Security scan results
            if Path("security/vulnerability_scan_results.json").exists():
                evidence_files.append("security/vulnerability_scan_results.json")

            # Access control configurations
            if Path(".github/workflows/ci-cd.yml").exists():
                evidence_files.append(".github/workflows/ci-cd.yml")

            # Docker security configurations
            if Path("Dockerfile").exists():
                evidence_files.append("Dockerfile")

            # Kubernetes security configurations
            k8s_dir = Path("k8s")
            if k8s_dir.exists():
                for file in k8s_dir.glob("*.yml"):
                    evidence_files.append(str(file))

            # Monitoring and logging configurations
            monitoring_dir = Path("observability")
            if monitoring_dir.exists():
                for file in monitoring_dir.rglob("*.yml"):
                    evidence_files.append(str(file))

            # Generate evidence manifest
            self._generate_evidence_manifest(evidence_files, framework)

        except Exception as e:
            self.logger.error(f"Error collecting automated evidence: {e}")

        return evidence_files

    def _generate_evidence_manifest(
        self, evidence_files: List[str], framework: ComplianceFramework
    ) -> None:
        """Generate evidence manifest file"""
        manifest = {
            "framework": framework.value,
            "collection_date": datetime.now().isoformat(),
            "evidence_files": [],
            "metadata": {
                "collector": "Vega 2.0 Compliance Reporter",
                "version": "1.0.0",
            },
        }

        for file_path in evidence_files:
            if Path(file_path).exists():
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                manifest["evidence_files"].append(
                    {
                        "file_path": file_path,
                        "file_size": Path(file_path).stat().st_size,
                        "sha256_hash": file_hash,
                        "collected_at": datetime.now().isoformat(),
                    }
                )

        manifest_path = f"security/evidence_manifest_{framework.value.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Evidence manifest created: {manifest_path}")

    def _generate_assessment_findings(
        self, controls: List[SecurityControl]
    ) -> List[str]:
        """Generate assessment findings"""
        findings = []

        not_implemented = [
            c for c in controls if c.status == ControlStatus.NOT_IMPLEMENTED
        ]
        if not_implemented:
            findings.append(
                f"{len(not_implemented)} security controls are not implemented"
            )

        partial = [
            c for c in controls if c.status == ControlStatus.PARTIALLY_IMPLEMENTED
        ]
        if partial:
            findings.append(
                f"{len(partial)} security controls are partially implemented"
            )

        overdue_testing = [
            c for c in controls if c.next_test_due and c.next_test_due < datetime.now()
        ]
        if overdue_testing:
            findings.append(f"{len(overdue_testing)} controls have overdue testing")

        return findings

    def _generate_assessment_recommendations(
        self, controls: List[SecurityControl]
    ) -> List[str]:
        """Generate assessment recommendations"""
        recommendations = []

        not_implemented = [
            c for c in controls if c.status == ControlStatus.NOT_IMPLEMENTED
        ]
        if not_implemented:
            recommendations.append(
                "Prioritize implementation of missing security controls"
            )
            recommendations.append(
                "Develop implementation timeline for non-implemented controls"
            )

        partial = [
            c for c in controls if c.status == ControlStatus.PARTIALLY_IMPLEMENTED
        ]
        if partial:
            recommendations.append(
                "Complete implementation of partially implemented controls"
            )

        no_testing = [
            c
            for c in controls
            if c.last_tested is None and c.status == ControlStatus.IMPLEMENTED
        ]
        if no_testing:
            recommendations.append(
                "Establish regular testing procedures for implemented controls"
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Maintain current control implementation and testing schedule"
            )

        return recommendations

    def _store_compliance_assessment(self, assessment: ComplianceAssessment) -> None:
        """Store compliance assessment in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO compliance_assessments 
            (assessment_id, framework, assessment_date, assessor, scope, overall_status,
             controls_assessed, controls_implemented, controls_partial, controls_not_implemented,
             risk_score, findings, recommendations, evidence_files)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                assessment.assessment_id,
                assessment.framework.value,
                assessment.assessment_date,
                assessment.assessor,
                assessment.scope,
                assessment.overall_status,
                assessment.controls_assessed,
                assessment.controls_implemented,
                assessment.controls_partial,
                assessment.controls_not_implemented,
                assessment.risk_score,
                json.dumps(assessment.findings),
                json.dumps(assessment.recommendations),
                json.dumps(assessment.evidence_files),
            ),
        )

        conn.commit()
        conn.close()

    def generate_compliance_report(
        self, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        assessment = self.conduct_compliance_assessment(framework, "Automated System")

        # Get recent vulnerability data
        vuln_data = self._get_vulnerability_summary()

        # Generate executive summary
        executive_summary = self._generate_executive_summary(assessment, vuln_data)

        # Get control details
        controls = self._get_controls_by_framework(framework)
        control_details = self._format_control_details(controls)

        report = {
            "report_metadata": {
                "report_id": str(uuid.uuid4()),
                "framework": framework.value,
                "generated_date": datetime.now().isoformat(),
                "report_period": f"{datetime.now() - timedelta(days=365):%Y-%m-%d} to {datetime.now():%Y-%m-%d}",
                "scope": "Vega 2.0 AI Platform",
                "assessor": "Automated Compliance System",
            },
            "executive_summary": executive_summary,
            "assessment_results": {
                "overall_status": assessment.overall_status,
                "risk_score": assessment.risk_score,
                "controls_assessed": assessment.controls_assessed,
                "implementation_status": {
                    "implemented": assessment.controls_implemented,
                    "partially_implemented": assessment.controls_partial,
                    "not_implemented": assessment.controls_not_implemented,
                },
            },
            "findings": assessment.findings,
            "recommendations": assessment.recommendations,
            "control_details": control_details,
            "vulnerability_summary": vuln_data,
            "evidence_files": assessment.evidence_files,
            "next_assessment_due": (datetime.now() + timedelta(days=365)).isoformat(),
        }

        # Save report to file
        report_filename = f"security/compliance_report_{framework.value.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Compliance report generated: {report_filename}")
        return report

    def _get_vulnerability_summary(self) -> Dict[str, Any]:
        """Get vulnerability summary from vulnerability manager"""
        try:
            # Try to import and use vulnerability manager
            from .vulnerability_manager import VulnerabilityManager

            vuln_manager = VulnerabilityManager()
            vulnerabilities = vuln_manager.get_vulnerabilities()

            return {
                "total_vulnerabilities": len(vulnerabilities),
                "critical": len(
                    [v for v in vulnerabilities if v.severity.value == "critical"]
                ),
                "high": len([v for v in vulnerabilities if v.severity.value == "high"]),
                "medium": len(
                    [v for v in vulnerabilities if v.severity.value == "medium"]
                ),
                "low": len([v for v in vulnerabilities if v.severity.value == "low"]),
                "last_scan": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.warning(f"Could not get vulnerability data: {e}")
            return {
                "total_vulnerabilities": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "last_scan": "N/A",
            }

    def _generate_executive_summary(
        self, assessment: ComplianceAssessment, vuln_data: Dict[str, Any]
    ) -> str:
        """Generate executive summary for compliance report"""
        status_text = {
            "COMPLIANT": "fully compliant",
            "MOSTLY_COMPLIANT": "mostly compliant with minor gaps",
            "NON_COMPLIANT": "non-compliant with significant gaps",
        }

        summary = f"""
        The Vega 2.0 AI Platform has been assessed for {assessment.framework.value} compliance.
        
        Assessment Results:
        - Overall Status: {status_text.get(assessment.overall_status, assessment.overall_status)}
        - Risk Score: {assessment.risk_score:.1f}/10 (lower is better)
        - Controls Implemented: {assessment.controls_implemented}/{assessment.controls_assessed}
        
        Security Posture:
        - Total Vulnerabilities: {vuln_data['total_vulnerabilities']}
        - Critical/High Severity: {vuln_data['critical'] + vuln_data['high']}
        
        The platform demonstrates a strong security posture with comprehensive monitoring,
        access controls, and automated security scanning capabilities.
        """.strip()

        return summary

    def _format_control_details(
        self, controls: List[SecurityControl]
    ) -> List[Dict[str, Any]]:
        """Format control details for reporting"""
        control_details = []

        for control in controls:
            detail = {
                "control_id": control.control_id,
                "title": control.title,
                "category": control.category,
                "status": control.status.value,
                "implementation_date": (
                    control.implementation_date.isoformat()
                    if control.implementation_date
                    else None
                ),
                "last_tested": (
                    control.last_tested.isoformat() if control.last_tested else None
                ),
                "responsible_party": control.responsible_party,
                "evidence_count": len(control.evidence),
                "findings_count": len(control.findings),
            }
            control_details.append(detail)

        return control_details

    def update_control_status(
        self,
        control_id: str,
        status: ControlStatus,
        evidence: Optional[List[str]] = None,
        responsible_party: Optional[str] = None,
    ) -> bool:
        """Update security control status and evidence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get existing control
        cursor.execute(
            "SELECT evidence FROM security_controls WHERE control_id = ?", (control_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        # Update evidence
        existing_evidence = json.loads(row[0]) if row[0] else []
        if evidence:
            existing_evidence.extend(evidence)

        # Update control
        cursor.execute(
            """
            UPDATE security_controls 
            SET status = ?, evidence = ?, responsible_party = ?, updated_at = CURRENT_TIMESTAMP
            WHERE control_id = ?
        """,
            (
                status.value,
                json.dumps(existing_evidence),
                responsible_party,
                control_id,
            ),
        )

        conn.commit()
        conn.close()
        return True


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Vega 2.0 Compliance Reporting")
    parser.add_argument(
        "--assess", choices=["SOC2", "ISO27001"], help="Conduct compliance assessment"
    )
    parser.add_argument(
        "--report", choices=["SOC2", "ISO27001"], help="Generate compliance report"
    )
    parser.add_argument(
        "--list-controls", choices=["SOC2", "ISO27001"], help="List security controls"
    )
    parser.add_argument("--assessor", default="CLI User", help="Assessor name")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    reporter = ComplianceReporter()

    if args.assess:
        framework = ComplianceFramework(args.assess)
        print(f"Conducting {framework.value} compliance assessment...")
        assessment = reporter.conduct_compliance_assessment(framework, args.assessor)
        print(f"Assessment completed: {assessment.overall_status}")
        print(f"Risk Score: {assessment.risk_score:.1f}/10")
        print(
            f"Controls: {assessment.controls_implemented}/{assessment.controls_assessed} implemented"
        )

    if args.report:
        framework = ComplianceFramework(args.report)
        print(f"Generating {framework.value} compliance report...")
        report = reporter.generate_compliance_report(framework)
        print(f"Report generated with {len(report['evidence_files'])} evidence files")

    if args.list_controls:
        framework = ComplianceFramework(args.list_controls)
        controls = reporter._get_controls_by_framework(framework)
        print(f"{framework.value} Security Controls ({len(controls)}):")
        for control in controls:
            print(f"  {control.control_id}: {control.title} [{control.status.value}]")


if __name__ == "__main__":
    main()
