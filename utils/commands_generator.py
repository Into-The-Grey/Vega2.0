from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "core" / "cli.py"


def _collect_cli_commands(cli_source: str) -> Dict[str, List[str]]:
    """Parse Typer CLI source and collect commands by group.

    Returns mapping: group -> list of command names. Root group key is "root".
    """
    tree = ast.parse(cli_source)
    # Map sub-app variable name -> group name used in app.add_typer(..., name="group")
    group_var_to_name: Dict[str, str] = {}

    class AddTyperVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # Look for app.add_typer(<var>, name="...")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_typer" and isinstance(
                    node.func.value, ast.Name
                ):
                    # get first arg var name
                    if node.args and isinstance(node.args[0], ast.Name):
                        var = node.args[0].id
                        group_name = None
                        # find keyword name="..."
                        for kw in node.keywords or []:
                            if (
                                kw.arg == "name"
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                            ):
                                group_name = kw.value.value
                                break
                        if group_name:
                            group_var_to_name[var] = group_name
            self.generic_visit(node)

    AddTyperVisitor().visit(tree)

    # Collect commands from decorators like @app.command(...) or @<group_var>.command("name")
    commands: Dict[str, List[str]] = {"root": []}

    class CommandVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            for dec in node.decorator_list:
                # Looking for Call(Attribute(Name(id=app|group_var), attr=command))
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr == "command" and isinstance(
                        dec.func.value, ast.Name
                    ):
                        var = dec.func.value.id  # e.g., app, search_app
                        # Command name: explicit string arg or fallback to function name
                        cmd_name = node.name
                        # Positional name
                        if (
                            dec.args
                            and isinstance(dec.args[0], ast.Constant)
                            and isinstance(dec.args[0].value, str)
                        ):
                            cmd_name = dec.args[0].value
                        # Keyword name="..."
                        for kw in dec.keywords or []:
                            if (
                                kw.arg == "name"
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                            ):
                                cmd_name = kw.value.value
                                break
                        # Group derivation
                        if var == "app":
                            group = "root"
                        else:
                            group = group_var_to_name.get(var) or var.replace(
                                "_app", ""
                            )
                        commands.setdefault(group, []).append(cmd_name)
            self.generic_visit(node)

    CommandVisitor().visit(tree)

    # Sort commands within groups
    for g in list(commands.keys()):
        commands[g] = sorted(sorted(set(commands[g])), key=str)
    return commands


def _collect_api_routes() -> List[Tuple[str, str]]:
    """Import FastAPI app and list (method, path) routes sorted."""
    # Ensure repo root on path and prefer core/ for potential name collisions
    core_dir = (REPO_ROOT / "core").as_posix()
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    root_dir = REPO_ROOT.as_posix()
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    from core.app import app  # type: ignore

    routes: List[Tuple[str, str]] = []
    for r in app.routes:
        path = getattr(r, "path", "")
        methods = sorted(getattr(r, "methods", []) or [])
        # Filter out docs, openapi, static
        if path.startswith("/static") or path in {
            "/openapi.json",
            "/docs",
            "/redoc",
            "/docs/oauth2-redirect",
        }:
            continue
        for m in methods:
            if m in {"HEAD", "OPTIONS"}:
                continue
            routes.append((m, path))
    routes = sorted(set(routes), key=lambda t: (t[1], t[0]))
    return routes


def generate_commands_markdown() -> str:
    """Generate dynamic commands markdown for CLI and API."""
    cli_src = CLI_PATH.read_text(encoding="utf-8")
    cli_cmds = _collect_cli_commands(cli_src)
    api_routes = _collect_api_routes()

    lines: List[str] = []
    lines.append("# Vega2.0 — Dynamic Commands\n")

    # CLI section
    lines.append("## CLI (auto-generated)\n")
    lines.append("```bash")
    # Root commands first
    for cmd in sorted(cli_cmds.get("root", [])):
        lines.append(f"python -m core.cli {cmd}")
    # Groups
    for group in sorted([g for g in cli_cmds.keys() if g != "root"]):
        for cmd in cli_cmds[group]:
            lines.append(f"python -m core.cli {group} {cmd}")
    lines.append("```\n")

    # API section
    lines.append("## API (auto-generated)\n")
    lines.append("```bash")
    lines.append('export BASE_URL="http://127.0.0.1:8000"')
    lines.append('export KEY="vega-default-key"')
    for method, path in api_routes:
        if method == "GET":
            lines.append(f"curl -s -H 'X-API-Key: $KEY' \"$BASE_URL{path}\"")
        else:
            # Default JSON content for POST
            lines.append(
                f"curl -s -X {method} -H 'X-API-Key: $KEY' -H 'Content-Type: application/json' \"$BASE_URL{path}\""
            )
    lines.append("```")

    return "\n".join(lines)


def generate_commands_html() -> str:
    """Wrap the markdown in a minimal HTML page using <pre> for simplicity."""
    md = generate_commands_markdown()
    escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Vega2.0 — Commands</title>
    <style>
      body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; }}
      pre {{ background: #0b1020; color: #e6edf3; padding: 16px; border-radius: 8px; overflow: auto; }}
      h1, h2 {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }}
      a {{ color: #58a6ff; text-decoration: none; }}
    </style>
  </head>
  <body>
    <h1>Vega2.0 — Dynamic Commands</h1>
    <p><a href="/">Home</a> · <a href="/static/index.html">Control Panel</a> · <a href="/docs/commands.md">View as Markdown</a></p>
    <pre>{escaped}</pre>
  </body>
  </html>
"""
