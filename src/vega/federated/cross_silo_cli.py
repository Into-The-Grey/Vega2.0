"""
Cross-Silo Federated Learning CLI

Command-line interface for managing organizations, silos, participants, and
cross-silo federated learning sessions.

Commands:
- org: Manage organizations
- silo: Manage silos within organizations
- participant: Manage participants
- session: Manage cross-silo federated learning sessions
- stats: View cross-organizational statistics
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cross_silo import OrganizationRole, FederationLevel
from cross_silo_coordinator import CrossSiloCoordinator, CrossSiloCoordinationConfig
from hierarchical_aggregation import LevelAggregationConfig

console = Console()


class CrossSiloCLI:
    """CLI for cross-silo federated learning management."""

    def __init__(self):
        """Initialize CLI."""
        self.coordinator = CrossSiloCoordinator()
        self.config_file = Path("cross_silo_config.json")
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config_data = json.load(f)
                self.coordinator.config = CrossSiloCoordinationConfig(**config_data)
                console.print(
                    f"[green]Loaded configuration from {self.config_file}[/green]"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.coordinator.config.to_dict(), f, indent=2)
            console.print(f"[green]Configuration saved to {self.config_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")


cli = CrossSiloCLI()


@click.group()
def main():
    """Cross-Silo Federated Learning Management CLI"""
    pass


# Organization Management Commands
@main.group()
def org():
    """Manage organizations"""
    pass


@org.command()
@click.option("--name", required=True, help="Organization name")
@click.option("--description", required=True, help="Organization description")
@click.option(
    "--domain", required=True, help="Organization domain (e.g., healthcare, finance)"
)
@click.option("--admin-contact", required=True, help="Admin contact email")
@click.option("--min-participants", default=2, help="Minimum participants")
@click.option("--max-participants", default=100, help="Maximum participants")
@click.option(
    "--data-sharing-policy",
    default="strict",
    type=click.Choice(["strict", "moderate", "open"]),
)
def create(
    name: str,
    description: str,
    domain: str,
    admin_contact: str,
    min_participants: int,
    max_participants: int,
    data_sharing_policy: str,
):
    """Create a new organization"""

    async def _create():
        try:
            org = await cli.coordinator.org_manager.create_organization(
                name=name,
                description=description,
                domain=domain,
                admin_contact=admin_contact,
                min_participants=min_participants,
                max_participants=max_participants,
                data_sharing_policy=data_sharing_policy,
            )
            console.print(
                f"[green]Created organization: {org.name} ({org.org_id})[/green]"
            )

            # Display organization details
            table = Table(title=f"Organization: {org.name}")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", org.org_id)
            table.add_row("Name", org.name)
            table.add_row("Domain", org.domain)
            table.add_row("Admin Contact", org.admin_contact)
            table.add_row("Data Sharing Policy", org.data_sharing_policy)
            table.add_row("Min Participants", str(org.min_participants))
            table.add_row("Max Participants", str(org.max_participants))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error creating organization: {e}[/red]")

    asyncio.run(_create())


@org.command()
def list():
    """List all organizations"""
    orgs = cli.coordinator.org_manager.organizations

    if not orgs:
        console.print("[yellow]No organizations found[/yellow]")
        return

    table = Table(title="Organizations")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Domain", style="green")
    table.add_column("Participants", style="blue")
    table.add_column("Silos", style="magenta")
    table.add_column("Status", style="yellow")

    for org in orgs.values():
        silo_count = len(cli.coordinator.org_manager.org_silos.get(org.org_id, set()))
        status = "Active" if org.is_active else "Inactive"

        table.add_row(
            org.org_id,
            org.name,
            org.domain,
            str(org.total_participants),
            str(silo_count),
            status,
        )

    console.print(table)


@org.command()
@click.argument("org_id")
def details(org_id: str):
    """Show detailed organization information"""
    hierarchy = cli.coordinator.org_manager.get_organization_hierarchy(org_id)

    if not hierarchy:
        console.print(f"[red]Organization {org_id} not found[/red]")
        return

    console.print(
        Panel(JSON.from_data(hierarchy), title=f"Organization Details: {org_id}")
    )


# Silo Management Commands
@main.group()
def silo():
    """Manage silos"""
    pass


@silo.command()
@click.option("--org-id", required=True, help="Organization ID")
@click.option("--name", required=True, help="Silo name")
@click.option("--description", required=True, help="Silo description")
@click.option(
    "--data-type", required=True, help="Data type (e.g., tabular, image, text)"
)
@click.option("--location", required=True, help="Silo location")
@click.option("--contact", required=True, help="Contact email")
@click.option(
    "--compute-capacity", default="medium", type=click.Choice(["low", "medium", "high"])
)
@click.option(
    "--privacy-level", default="high", type=click.Choice(["low", "medium", "high"])
)
def create(
    org_id: str,
    name: str,
    description: str,
    data_type: str,
    location: str,
    contact: str,
    compute_capacity: str,
    privacy_level: str,
):
    """Create a new silo"""

    async def _create():
        try:
            silo = await cli.coordinator.org_manager.create_silo(
                org_id=org_id,
                name=name,
                description=description,
                data_type=data_type,
                location=location,
                contact=contact,
                compute_capacity=compute_capacity,
                privacy_level=privacy_level,
            )
            console.print(f"[green]Created silo: {silo.name} ({silo.silo_id})[/green]")

            # Display silo details
            table = Table(title=f"Silo: {silo.name}")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", silo.silo_id)
            table.add_row("Organization", org_id)
            table.add_row("Name", silo.name)
            table.add_row("Data Type", silo.data_type)
            table.add_row("Location", silo.location)
            table.add_row("Compute Capacity", silo.compute_capacity)
            table.add_row("Privacy Level", silo.privacy_level)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error creating silo: {e}[/red]")

    asyncio.run(_create())


@silo.command()
@click.option("--org-id", help="Filter by organization ID")
def list(org_id: Optional[str]):
    """List all silos"""
    silos = cli.coordinator.org_manager.silos

    if org_id:
        silos = {k: v for k, v in silos.items() if v.org_id == org_id}

    if not silos:
        console.print("[yellow]No silos found[/yellow]")
        return

    table = Table(title="Silos")
    table.add_column("ID", style="cyan")
    table.add_column("Organization", style="blue")
    table.add_column("Name", style="white")
    table.add_column("Data Type", style="green")
    table.add_column("Location", style="magenta")
    table.add_column("Participants", style="yellow")
    table.add_column("Status", style="red")

    for silo in silos.values():
        participant_count = len(
            cli.coordinator.org_manager.silo_participants.get(silo.silo_id, set())
        )
        status = "Active" if silo.is_active else "Inactive"

        table.add_row(
            silo.silo_id,
            silo.org_id,
            silo.name,
            silo.data_type,
            silo.location,
            str(participant_count),
            status,
        )

    console.print(table)


# Participant Management Commands
@main.group()
def participant():
    """Manage participants"""
    pass


@participant.command()
@click.option("--org-id", required=True, help="Organization ID")
@click.option("--silo-id", help="Silo ID (optional)")
@click.option("--name", required=True, help="Participant name")
@click.option(
    "--role",
    required=True,
    type=click.Choice(["admin", "coordinator", "participant", "observer"]),
)
@click.option("--contact", required=True, help="Contact email")
def register(org_id: str, silo_id: Optional[str], name: str, role: str, contact: str):
    """Register a new participant"""

    async def _register():
        try:
            participant = await cli.coordinator.org_manager.register_participant(
                org_id=org_id,
                silo_id=silo_id,
                name=name,
                role=OrganizationRole(role),
                contact=contact,
            )
            console.print(
                f"[green]Registered participant: {participant.name} ({participant.participant_id})[/green]"
            )

            # Display participant details
            table = Table(title=f"Participant: {participant.name}")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", participant.participant_id)
            table.add_row("Organization", participant.org_id)
            table.add_row("Silo", participant.silo_id or "None")
            table.add_row("Name", participant.name)
            table.add_row("Role", participant.role.value)
            table.add_row("Contact", participant.contact)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error registering participant: {e}[/red]")

    asyncio.run(_register())


@participant.command()
@click.option("--org-id", help="Filter by organization ID")
@click.option("--silo-id", help="Filter by silo ID")
def list(org_id: Optional[str], silo_id: Optional[str]):
    """List all participants"""
    participants = cli.coordinator.org_manager.participants

    if org_id:
        participants = {k: v for k, v in participants.items() if v.org_id == org_id}

    if silo_id:
        participants = {k: v for k, v in participants.items() if v.silo_id == silo_id}

    if not participants:
        console.print("[yellow]No participants found[/yellow]")
        return

    table = Table(title="Participants")
    table.add_column("ID", style="cyan")
    table.add_column("Organization", style="blue")
    table.add_column("Silo", style="green")
    table.add_column("Name", style="white")
    table.add_column("Role", style="magenta")
    table.add_column("Contact", style="yellow")
    table.add_column("Status", style="red")

    for participant in participants.values():
        status = "Active" if participant.is_active else "Inactive"

        table.add_row(
            participant.participant_id,
            participant.org_id,
            participant.silo_id or "None",
            participant.name,
            participant.role.value,
            participant.contact,
            status,
        )

    console.print(table)


# Session Management Commands
@main.group()
def session():
    """Manage cross-silo federated learning sessions"""
    pass


@session.command()
@click.option("--name", required=True, help="Session name")
@click.option("--description", required=True, help="Session description")
@click.option(
    "--orgs", required=True, help="Participating organization IDs (comma-separated)"
)
@click.option("--silos", help="Participating silos as org_id:silo_id,silo_id format")
def create(name: str, description: str, orgs: str, silos: Optional[str]):
    """Create a new cross-silo federated learning session"""

    async def _create():
        try:
            # Parse participating organizations
            participating_orgs = [org.strip() for org in orgs.split(",")]

            # Parse participating silos
            participating_silos = {}
            if silos:
                for silo_spec in silos.split(","):
                    if ":" in silo_spec:
                        org_id, silo_list = silo_spec.split(":", 1)
                        org_id = org_id.strip()
                        silo_ids = [s.strip() for s in silo_list.split("|")]
                        participating_silos[org_id] = silo_ids
            else:
                # Auto-assign all silos from participating organizations
                for org_id in participating_orgs:
                    org_silos = cli.coordinator.org_manager.org_silos.get(org_id, set())
                    if org_silos:
                        participating_silos[org_id] = list(org_silos)

            session = await cli.coordinator.create_cross_silo_session(
                name=name,
                description=description,
                participating_orgs=participating_orgs,
                participating_silos=participating_silos,
            )

            console.print(
                f"[green]Created session: {session.name} ({session.session_id})[/green]"
            )

            # Display session details
            table = Table(title=f"Session: {session.name}")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", session.session_id)
            table.add_row("Name", session.name)
            table.add_row("Description", session.description)
            table.add_row("Organizations", ", ".join(session.participating_orgs))
            table.add_row("Status", session.status.value)
            table.add_row("Current Round", str(session.current_round))
            table.add_row("Max Rounds", str(session.max_rounds))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error creating session: {e}[/red]")

    asyncio.run(_create())


@session.command()
def list():
    """List all sessions"""
    sessions = cli.coordinator.active_sessions

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return

    table = Table(title="Cross-Silo Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Organizations", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Round", style="magenta")
    table.add_column("Created", style="yellow")

    for session in sessions.values():
        created_time = session.created_at

        table.add_row(
            session.session_id,
            session.name,
            str(len(session.participating_orgs)),
            session.status.value,
            f"{session.current_round}/{session.max_rounds}",
            f"{created_time:.0f}",
        )

    console.print(table)


@session.command()
@click.argument("session_id")
def status(session_id: str):
    """Show detailed session status"""
    status_data = cli.coordinator.get_session_status(session_id)

    if "error" in status_data:
        console.print(f"[red]{status_data['error']}[/red]")
        return

    console.print(
        Panel(JSON.from_data(status_data), title=f"Session Status: {session_id}")
    )


@session.command()
@click.argument("session_id")
def start_round(session_id: str):
    """Start a new federation round"""

    async def _start():
        try:
            round_state = await cli.coordinator.start_federated_round(session_id)
            console.print(
                f"[green]Started round {round_state.round_number} for session {session_id}[/green]"
            )

        except Exception as e:
            console.print(f"[red]Error starting round: {e}[/red]")

    asyncio.run(_start())


# Statistics Commands
@main.group()
def stats():
    """View cross-organizational statistics"""
    pass


@stats.command()
def overview():
    """Show comprehensive statistics overview"""
    stats_data = cli.coordinator.get_cross_org_statistics()
    console.print(
        Panel(JSON.from_data(stats_data), title="Cross-Organizational Statistics")
    )


@stats.command()
def aggregation():
    """Show aggregation performance statistics"""
    agg_stats = cli.coordinator.hierarchical_aggregator.get_aggregation_statistics()
    console.print(
        Panel(JSON.from_data(agg_stats), title="Aggregation Performance Statistics")
    )


# Configuration Commands
@main.group()
def config():
    """Manage configuration"""
    pass


@config.command()
def show():
    """Show current configuration"""
    config_data = cli.coordinator.config.to_dict()
    console.print(Panel(JSON.from_data(config_data), title="Current Configuration"))


@config.command()
@click.option(
    "--min-orgs", type=int, help="Minimum organizations for global aggregation"
)
@click.option("--min-silos", type=int, help="Minimum silos per organization")
@click.option("--min-participants", type=int, help="Minimum participants per silo")
@click.option("--max-rounds", type=int, help="Maximum federation rounds")
@click.option("--convergence-threshold", type=float, help="Convergence threshold")
def update(
    min_orgs: Optional[int],
    min_silos: Optional[int],
    min_participants: Optional[int],
    max_rounds: Optional[int],
    convergence_threshold: Optional[float],
):
    """Update configuration"""
    if min_orgs is not None:
        cli.coordinator.config.min_organizations = min_orgs
    if min_silos is not None:
        cli.coordinator.config.min_silos_per_org = min_silos
    if min_participants is not None:
        cli.coordinator.config.min_participants_per_silo = min_participants
    if max_rounds is not None:
        cli.coordinator.config.max_rounds = max_rounds
    if convergence_threshold is not None:
        cli.coordinator.config.convergence_threshold = convergence_threshold

    cli.save_config()
    console.print("[green]Configuration updated successfully[/green]")


if __name__ == "__main__":
    main()
