"""
Security Integration System
Orchestrates all security scanning, vulnerability management, and compliance reporting
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import subprocess
import yaml

from .scanner import SecurityScanner
from .vuln_manager import VulnerabilityManager
from .compliance import ComplianceReporter

logger = logging.getLogger(__name__)


class SecurityOrchestrator:
    """Orchestrates all security operations for Vega2.0"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.scanner = SecurityScanner(self.config.get("scanner", {}))
        self.vuln_manager = VulnerabilityManager(self.config.get("vulnerability", {}))
        self.compliance = ComplianceReporter(self.config.get("compliance", {}))
        self.results_dir = Path(self.config.get("results_dir", "security/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load security configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            "scanner": {
                "enabled_tools": ["bandit", "safety", "semgrep"],
                "severity_threshold": "medium",
                "fail_on_high": True,
            },
            "vulnerability": {
                "auto_fix": False,
                "notify_on_critical": True,
                "max_age_days": 30,
            },
            "compliance": {
                "frameworks": ["SOC2", "ISO27001", "GDPR"],
                "generate_reports": True,
                "schedule": "weekly",
            },
            "results_dir": "security/results",
        }

    async def run_full_security_audit(self) -> Dict[str, Any]:
        """Run complete security audit including scanning, vulnerability assessment, and compliance check"""
        logger.info("Starting full security audit")
        timestamp = datetime.now().isoformat()

        results = {
            "timestamp": timestamp,
            "audit_id": f"audit_{timestamp.replace(':', '-')}",
            "status": "running",
        }

        try:
            # 1. Security scanning
            logger.info("Running security scans...")
            scan_results = await self.scanner.run_comprehensive_scan()
            results["security_scan"] = scan_results

            # 2. Vulnerability management
            logger.info("Assessing vulnerabilities...")
            vuln_results = await self.vuln_manager.assess_all_vulnerabilities()
            results["vulnerabilities"] = vuln_results

            # 3. Compliance reporting
            logger.info("Generating compliance reports...")
            compliance_results = await self.compliance.generate_all_reports()
            results["compliance"] = compliance_results

            # 4. Generate summary
            results["summary"] = self._generate_audit_summary(results)
            results["status"] = "completed"

            # Save results
            await self._save_audit_results(results)

            logger.info(f"Security audit completed. Audit ID: {results['audit_id']}")
            return results

        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    def _generate_audit_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of security audit"""
        summary = {
            "overall_status": "pass",
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "recommendations": [],
        }

        # Analyze security scan results
        if "security_scan" in results:
            scan_summary = results["security_scan"].get("summary", {})
            summary["critical_issues"] += scan_summary.get("critical", 0)
            summary["high_issues"] += scan_summary.get("high", 0)
            summary["medium_issues"] += scan_summary.get("medium", 0)
            summary["low_issues"] += scan_summary.get("low", 0)

        # Analyze vulnerability results
        if "vulnerabilities" in results:
            vuln_summary = results["vulnerabilities"].get("summary", {})
            summary["critical_issues"] += vuln_summary.get("critical", 0)
            summary["high_issues"] += vuln_summary.get("high", 0)

        # Analyze compliance results
        if "compliance" in results:
            compliance_summary = results["compliance"].get("summary", {})
            if compliance_summary.get("non_compliant_controls", 0) > 0:
                summary["recommendations"].append(
                    "Address compliance gaps identified in report"
                )

        # Determine overall status
        if summary["critical_issues"] > 0:
            summary["overall_status"] = "critical"
            summary["recommendations"].insert(
                0, "Critical security issues require immediate attention"
            )
        elif summary["high_issues"] > 3:
            summary["overall_status"] = "warning"
            summary["recommendations"].insert(
                0, "Multiple high-severity issues should be addressed promptly"
            )

        return summary

    async def _save_audit_results(self, results: Dict[str, Any]):
        """Save audit results to file"""
        audit_file = self.results_dir / f"{results['audit_id']}.json"
        with open(audit_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Also save latest results
        latest_file = self.results_dir / "latest_audit.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    async def run_ci_security_check(self) -> bool:
        """Run security checks for CI/CD pipeline"""
        logger.info("Running CI security checks")

        try:
            # Quick security scan
            scan_results = await self.scanner.run_quick_scan()

            # Check for critical/high severity issues
            summary = scan_results.get("summary", {})
            critical = summary.get("critical", 0)
            high = summary.get("high", 0)

            # Fail CI if critical issues or too many high issues
            fail_threshold = self.config.get("scanner", {}).get("fail_on_high", True)

            if critical > 0:
                logger.error(f"CI failed: {critical} critical security issues found")
                return False

            if fail_threshold and high > 2:
                logger.error(f"CI failed: {high} high-severity security issues found")
                return False

            logger.info("CI security checks passed")
            return True

        except Exception as e:
            logger.error(f"CI security check failed: {e}")
            return False

    async def monitor_security_status(self) -> Dict[str, Any]:
        """Monitor ongoing security status"""
        try:
            # Get latest vulnerability status
            vuln_status = await self.vuln_manager.get_current_status()

            # Get compliance status
            compliance_status = await self.compliance.get_current_status()

            # Get recent scan results
            latest_scan = self.results_dir / "latest_audit.json"
            scan_status = {}
            if latest_scan.exists():
                with open(latest_scan) as f:
                    data = json.load(f)
                    scan_status = data.get("summary", {})

            return {
                "timestamp": datetime.now().isoformat(),
                "vulnerabilities": vuln_status,
                "compliance": compliance_status,
                "latest_scan": scan_status,
                "overall_health": self._calculate_security_health(
                    vuln_status, compliance_status, scan_status
                ),
            }

        except Exception as e:
            logger.error(f"Security monitoring failed: {e}")
            return {"error": str(e)}

    def _calculate_security_health(
        self, vuln_status: Dict, compliance_status: Dict, scan_status: Dict
    ) -> str:
        """Calculate overall security health score"""
        score = 100

        # Deduct for vulnerabilities
        score -= vuln_status.get("critical", 0) * 20
        score -= vuln_status.get("high", 0) * 10
        score -= vuln_status.get("medium", 0) * 5

        # Deduct for compliance gaps
        score -= compliance_status.get("non_compliant_controls", 0) * 15

        # Deduct for scan issues
        score -= scan_status.get("critical_issues", 0) * 15
        score -= scan_status.get("high_issues", 0) * 8

        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"


# CLI interface for security operations
async def main():
    """Main CLI entry point"""
    import sys

    orchestrator = SecurityOrchestrator()

    if len(sys.argv) < 2:
        print("Usage: python -m vega.security.integration <command>")
        print("Commands: audit, ci-check, monitor, status")
        return

    command = sys.argv[1]

    if command == "audit":
        results = await orchestrator.run_full_security_audit()
        print(f"Audit completed: {results['audit_id']}")
        print(f"Status: {results['status']}")
        if "summary" in results:
            summary = results["summary"]
            print(f"Overall status: {summary['overall_status']}")
            print(f"Critical issues: {summary['critical_issues']}")
            print(f"High issues: {summary['high_issues']}")

    elif command == "ci-check":
        success = await orchestrator.run_ci_security_check()
        sys.exit(0 if success else 1)

    elif command == "monitor":
        status = await orchestrator.monitor_security_status()
        print(json.dumps(status, indent=2, default=str))

    elif command == "status":
        status = await orchestrator.monitor_security_status()
        health = status.get("overall_health", "unknown")
        print(f"Security Health: {health}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())
