"""
Security CLI Commands for Vega2.0
Provides command-line interface for security operations
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Optional

from .integration import SecurityOrchestrator
from .scanner import SecurityScanner
from .vuln_manager import VulnerabilityManager
from .compliance import ComplianceReporter


@click.group()
def security():
    """Security management commands for Vega2.0"""
    pass


@security.command()
@click.option("--config", "-c", help="Security configuration file path")
@click.option("--output", "-o", help="Output file for results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "html", "text"]),
    default="json",
)
def audit(config: Optional[str], output: Optional[str], output_format: str):
    """Run comprehensive security audit"""

    async def run_audit():
        orchestrator = SecurityOrchestrator(config)
        results = await orchestrator.run_full_security_audit()

        if output_format == "json":
            output_data = json.dumps(results, indent=2, default=str)
        elif output_format == "text":
            output_data = format_text_report(results)
        else:  # html
            output_data = format_html_report(results)

        if output:
            with open(output, "w") as f:
                f.write(output_data)
            click.echo(f"Security audit results saved to {output}")
        else:
            click.echo(output_data)

        # Exit with error code if critical issues found
        summary = results.get("summary", {})
        if summary.get("critical_issues", 0) > 0:
            sys.exit(1)

    asyncio.run(run_audit())


@security.command()
@click.option("--config", "-c", help="Security configuration file path")
def scan(config: Optional[str]):
    """Run security vulnerability scan"""

    async def run_scan():
        scanner = SecurityScanner(config)
        results = await scanner.run_comprehensive_scan()

        click.echo("🔍 Security Scan Results")
        click.echo("=" * 50)

        summary = results.get("summary", {})
        click.echo(f"Critical: {summary.get('critical', 0)}")
        click.echo(f"High: {summary.get('high', 0)}")
        click.echo(f"Medium: {summary.get('medium', 0)}")
        click.echo(f"Low: {summary.get('low', 0)}")

        # Show critical and high issues
        tools = results.get("tools", {})
        for tool_name, tool_results in tools.items():
            issues = tool_results.get("issues", [])
            critical_high = [
                i for i in issues if i.get("severity") in ["CRITICAL", "HIGH"]
            ]

            if critical_high:
                click.echo(f"\n🚨 {tool_name.upper()} - Critical/High Issues:")
                for issue in critical_high[:5]:  # Show first 5
                    click.echo(
                        f"  • {issue.get('title', 'Unknown issue')} [{issue.get('severity')}]"
                    )
                if len(critical_high) > 5:
                    click.echo(f"  ... and {len(critical_high) - 5} more issues")

    asyncio.run(run_scan())


@security.command()
@click.option("--config", "-c", help="Security configuration file path")
def vulnerabilities(config: Optional[str]):
    """Manage security vulnerabilities"""

    async def run_vuln_check():
        vuln_manager = VulnerabilityManager(config)
        results = await vuln_manager.assess_all_vulnerabilities()

        click.echo("🔍 Vulnerability Assessment")
        click.echo("=" * 50)

        summary = results.get("summary", {})
        click.echo(f"Total vulnerabilities: {summary.get('total', 0)}")
        click.echo(f"Critical: {summary.get('critical', 0)}")
        click.echo(f"High: {summary.get('high', 0)}")
        click.echo(f"Medium: {summary.get('medium', 0)}")
        click.echo(f"Low: {summary.get('low', 0)}")

        # Show recent vulnerabilities
        vulnerabilities = results.get("vulnerabilities", [])
        recent_critical = [
            v for v in vulnerabilities if v.get("severity") == "CRITICAL"
        ][:3]

        if recent_critical:
            click.echo("\n🚨 Recent Critical Vulnerabilities:")
            for vuln in recent_critical:
                click.echo(
                    f"  • {vuln.get('id', 'Unknown')} - {vuln.get('title', 'No title')}"
                )
                click.echo(f"    Affected: {vuln.get('package', 'Unknown package')}")

    asyncio.run(run_vuln_check())


@security.command()
@click.option(
    "--framework",
    type=click.Choice(["SOC2", "ISO27001", "GDPR", "NIST", "all"]),
    default="all",
)
@click.option("--config", "-c", help="Security configuration file path")
@click.option("--output", "-o", help="Output directory for reports")
def compliance(framework: str, config: Optional[str], output: Optional[str]):
    """Generate compliance reports"""

    async def run_compliance():
        compliance_reporter = ComplianceReporter(config)

        if framework == "all":
            results = await compliance_reporter.generate_all_reports()
        else:
            results = await compliance_reporter.generate_framework_report(
                framework.lower()
            )

        click.echo(f"📋 Compliance Report - {framework}")
        click.echo("=" * 50)

        if framework == "all":
            for fw, report in results.items():
                summary = report.get("summary", {})
                compliant = summary.get("compliant_controls", 0)
                total = summary.get("total_controls", 0)
                percentage = (compliant / total * 100) if total > 0 else 0

                click.echo(
                    f"{fw.upper()}: {compliant}/{total} controls ({percentage:.1f}%)"
                )
        else:
            summary = results.get("summary", {})
            compliant = summary.get("compliant_controls", 0)
            total = summary.get("total_controls", 0)
            percentage = (compliant / total * 100) if total > 0 else 0

            click.echo(f"Compliant controls: {compliant}/{total} ({percentage:.1f}%)")

            gaps = results.get("gaps", [])
            if gaps:
                click.echo("\n⚠️ Compliance Gaps:")
                for gap in gaps[:5]:
                    click.echo(
                        f"  • {gap.get('control', 'Unknown')} - {gap.get('description', 'No description')}"
                    )

        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)

            if framework == "all":
                for fw, report in results.items():
                    report_file = output_dir / f"{fw.lower()}_compliance_report.json"
                    with open(report_file, "w") as f:
                        json.dump(report, f, indent=2, default=str)
            else:
                report_file = output_dir / f"{framework.lower()}_compliance_report.json"
                with open(report_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)

            click.echo(f"\nReports saved to {output}")

    asyncio.run(run_compliance())


@security.command()
@click.option("--config", "-c", help="Security configuration file path")
def monitor(config: Optional[str]):
    """Monitor security status"""

    async def run_monitor():
        orchestrator = SecurityOrchestrator(config)
        status = await orchestrator.monitor_security_status()

        click.echo("📊 Security Status Monitor")
        click.echo("=" * 50)

        health = status.get("overall_health", "unknown")
        health_emoji = {
            "excellent": "🟢",
            "good": "🟡",
            "fair": "🟠",
            "poor": "🔴",
            "critical": "🚨",
        }.get(health, "❓")

        click.echo(f"Overall Health: {health_emoji} {health.upper()}")

        # Vulnerability status
        vuln_status = status.get("vulnerabilities", {})
        if vuln_status:
            click.echo(f"\n🔍 Vulnerabilities:")
            click.echo(f"  Critical: {vuln_status.get('critical', 0)}")
            click.echo(f"  High: {vuln_status.get('high', 0)}")
            click.echo(f"  Medium: {vuln_status.get('medium', 0)}")

        # Compliance status
        compliance_status = status.get("compliance", {})
        if compliance_status:
            click.echo(f"\n📋 Compliance:")
            click.echo(
                f"  Non-compliant controls: {compliance_status.get('non_compliant_controls', 0)}"
            )

        # Latest scan
        scan_status = status.get("latest_scan", {})
        if scan_status:
            click.echo(f"\n🔍 Latest Scan:")
            click.echo(f"  Critical issues: {scan_status.get('critical_issues', 0)}")
            click.echo(f"  High issues: {scan_status.get('high_issues', 0)}")

    asyncio.run(run_monitor())


@security.command()
@click.option("--config", "-c", help="Security configuration file path")
def ci_check(config: Optional[str]):
    """Run security checks for CI/CD pipeline"""

    async def run_ci_check():
        orchestrator = SecurityOrchestrator(config)
        success = await orchestrator.run_ci_security_check()

        if success:
            click.echo("✅ CI security checks passed")
            sys.exit(0)
        else:
            click.echo("❌ CI security checks failed")
            sys.exit(1)

    asyncio.run(run_ci_check())


@security.command()
@click.option("--days", type=int, default=7, help="Number of days to look back")
def history(days: int):
    """Show security audit history"""
    results_dir = Path("security/results")

    if not results_dir.exists():
        click.echo("No security results found")
        return

    audit_files = sorted(results_dir.glob("audit_*.json"), reverse=True)

    if not audit_files:
        click.echo("No audit history found")
        return

    click.echo("📈 Security Audit History")
    click.echo("=" * 50)

    for audit_file in audit_files[:days]:
        try:
            with open(audit_file) as f:
                data = json.load(f)

            timestamp = data.get("timestamp", "Unknown")
            status = data.get("status", "unknown")
            summary = data.get("summary", {})

            status_emoji = "✅" if status == "completed" else "❌"

            click.echo(f"{status_emoji} {timestamp}")
            click.echo(
                f"   Critical: {summary.get('critical_issues', 0)}, "
                f"High: {summary.get('high_issues', 0)}, "
                f"Status: {summary.get('overall_status', 'unknown')}"
            )

        except Exception as e:
            click.echo(f"❌ Error reading {audit_file}: {e}")


def format_text_report(results: dict) -> str:
    """Format audit results as text report"""
    lines = []
    lines.append("🔒 VEGA 2.0 SECURITY AUDIT REPORT")
    lines.append("=" * 50)
    lines.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    lines.append(f"Audit ID: {results.get('audit_id', 'Unknown')}")
    lines.append(f"Status: {results.get('status', 'Unknown')}")
    lines.append("")

    summary = results.get("summary", {})
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 20)
    lines.append(f"Overall Status: {summary.get('overall_status', 'Unknown')}")
    lines.append(f"Critical Issues: {summary.get('critical_issues', 0)}")
    lines.append(f"High Issues: {summary.get('high_issues', 0)}")
    lines.append(f"Medium Issues: {summary.get('medium_issues', 0)}")
    lines.append(f"Low Issues: {summary.get('low_issues', 0)}")
    lines.append("")

    recommendations = summary.get("recommendations", [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 15)
        for rec in recommendations:
            lines.append(f"• {rec}")
        lines.append("")

    return "\n".join(lines)


def format_html_report(results: dict) -> str:
    """Format audit results as HTML report"""
    summary = results.get("summary", {})

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vega 2.0 Security Audit Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #2d3748; color: white; padding: 20px; border-radius: 5px; }}
            .summary {{ background: #f7fafc; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            .critical {{ color: #e53e3e; }}
            .high {{ color: #dd6b20; }}
            .medium {{ color: #d69e2e; }}
            .low {{ color: #38a169; }}
            .status-pass {{ color: #38a169; }}
            .status-warning {{ color: #d69e2e; }}
            .status-critical {{ color: #e53e3e; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔒 Vega 2.0 Security Audit Report</h1>
            <p>Audit ID: {results.get('audit_id', 'Unknown')}</p>
            <p>Generated: {results.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Overall Status:</strong> 
                <span class="status-{summary.get('overall_status', 'unknown').lower()}">
                    {summary.get('overall_status', 'Unknown').upper()}
                </span>
            </p>
            
            <h3>Issue Summary</h3>
            <ul>
                <li class="critical">Critical Issues: {summary.get('critical_issues', 0)}</li>
                <li class="high">High Issues: {summary.get('high_issues', 0)}</li>
                <li class="medium">Medium Issues: {summary.get('medium_issues', 0)}</li>
                <li class="low">Low Issues: {summary.get('low_issues', 0)}</li>
            </ul>
        </div>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    security()
