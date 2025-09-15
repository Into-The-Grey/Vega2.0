"""
cli.py - Typer + Rich CLI for Vega2.0

Commands mirror API endpoints for parity:
- vega chat "Hello" -> prints model response
- vega repl -> interactive chat REPL with memory (session only)
- vega history --limit 20 -> pretty-print last N chats from SQLite
- vega integrations test -> send test Slack message (if webhook configured)
- vega dataset build ./myfiles -> run dataset preparation
- vega train --config training/config.yaml -> run fine-tuning

Usage:
  python -m cli --help
  Or add an entry point wrapper named 'vega' if desired.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Any

try:
    import typer  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # Minimal shim to keep module importable and provide a clear runtime message
    import sys

    class _TyperShim:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def add_typer(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            print(
                "Typer is required for the CLI. Install it with: pip install 'typer[all]'",
                file=sys.stderr,
            )
            sys.exit(1)

    class _TyperModule:
        Typer = _TyperShim

        @staticmethod
        def Option(default: "Any" = None, **kwargs: "Any") -> "Any":
            return default

        @staticmethod
        def Argument(default: "Any" = ..., **kwargs: "Any") -> "Any":
            return default

    typer = _TyperModule()

from rich.console import Console
from rich.table import Table

from config import get_config
from db import get_history, log_conversation
from llm import query_llm, LLMBackendError
from db import set_feedback
from integrations.slack_connector import send_slack_message

app = typer.Typer(help="Vega2.0 CLI")
integrations_app = typer.Typer(help="External integrations commands")
search_app = typer.Typer(help="Web and image search")
dataset_app = typer.Typer(help="Dataset utilities")
learn_app = typer.Typer(help="Learning and self-improvement")
db_app = typer.Typer(help="Database maintenance utilities")
gen_app = typer.Typer(help="Generation settings control")
osint_app = typer.Typer(help="OSINT utilities")
net_app = typer.Typer(help="Networking utilities")

app.add_typer(integrations_app, name="integrations")
app.add_typer(search_app, name="search")
app.add_typer(dataset_app, name="dataset")
app.add_typer(learn_app, name="learn")
app.add_typer(db_app, name="db")
app.add_typer(gen_app, name="gen")
app.add_typer(osint_app, name="osint")
app.add_typer(net_app, name="net")

# Add memory commands if available
try:
    from memory import MemoryManager

    memory_app = typer.Typer(help="Dynamic memory management")
    app.add_typer(memory_app, name="memory")

    # Define memory commands
    @memory_app.command("store")
    def memory_store(
        topic: str = typer.Argument(..., help="Topic for the knowledge item"),
        content: str = typer.Argument(..., help="Content to store"),
        key: str = typer.Option(
            None, help="Unique key (auto-generated if not provided)"
        ),
        metadata: str = typer.Option("{}", help="JSON metadata"),
        tags: str = typer.Option("", help="Comma-separated tags"),
    ):
        """Store a knowledge item in dynamic memory."""
        try:
            from memory import MemoryManager, MemoryItem

            manager = MemoryManager()

            # Parse metadata
            try:
                meta_dict = json.loads(metadata) if metadata.strip() else {}
            except json.JSONDecodeError:
                console.print("✗ Invalid JSON metadata", style="red")
                return

            # Add tags to metadata
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                meta_dict["tags"] = tag_list

            # Generate key if not provided
            if not key:
                import hashlib

                key = hashlib.md5(f"{topic}:{content[:100]}".encode()).hexdigest()[:8]

            # Create MemoryItem
            item = MemoryItem(
                key=key, topic=topic, content=content, metadata=meta_dict, source="cli"
            )

            success = manager.store_knowledge(item)

            if success:
                console.print(
                    f"✓ Stored knowledge item: {key} in topic '{topic}'", style="green"
                )
            else:
                console.print(f"✗ Failed to store knowledge item", style="red")

        except Exception as e:
            console.print(f"✗ Error storing knowledge: {e}", style="red")

    @memory_app.command("get")
    def memory_get(
        key: str = typer.Argument(..., help="Key of the knowledge item"),
        topic: str = typer.Argument(..., help="Topic of the knowledge item"),
    ):
        """Get a specific knowledge item by key and topic."""
        try:
            from memory import MemoryManager

            manager = MemoryManager()

            item = manager.get_knowledge(key, topic, source="cli")

            if item:
                table = Table(title=f"Knowledge Item: {key}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Key", item.key)
                table.add_row("Topic", item.topic)
                table.add_row("Source", item.source)
                table.add_row("Version", str(item.version))
                table.add_row("Usage Count", str(item.usage_count))
                table.add_row(
                    "Created", item.created_at.isoformat() if item.created_at else "N/A"
                )
                table.add_row(
                    "Updated", item.updated_at.isoformat() if item.updated_at else "N/A"
                )
                table.add_row(
                    "Last Used",
                    item.last_used_at.isoformat() if item.last_used_at else "N/A",
                )

                console.print(table)
                console.print(f"\n[bold]Content:[/bold]\n{item.content}")

                if item.metadata:
                    console.print(f"\n[bold]Metadata:[/bold]")
                    console.print(json.dumps(item.metadata, indent=2))
            else:
                console.print(
                    f"✗ Knowledge item not found: {key} in topic '{topic}'",
                    style="yellow",
                )

        except Exception as e:
            console.print(f"✗ Error getting knowledge: {e}", style="red")

    @memory_app.command("search")
    def memory_search(
        query: str = typer.Argument(..., help="Search query"),
        topic: str = typer.Option(None, help="Limit to specific topic"),
        limit: int = typer.Option(10, help="Maximum results to return"),
    ):
        """Search knowledge items by content."""
        try:
            from memory import MemoryManager

            manager = MemoryManager()

            results = manager.search_knowledge(
                query, topic=topic, limit=limit, source="cli"
            )

            if results:
                table = Table(
                    title=f"Search Results - '{query}'"
                    + (f" in '{topic}'" if topic else "")
                )
                table.add_column("Key", style="cyan")
                table.add_column("Topic", style="magenta")
                table.add_column("Usage", style="green")
                table.add_column("Last Used", style="yellow")
                table.add_column("Content Preview", max_width=50)

                for item in results:
                    last_used = (
                        item.last_used_at.strftime("%Y-%m-%d %H:%M")
                        if item.last_used_at
                        else "Never"
                    )
                    preview = (
                        item.content[:80] + "..."
                        if len(item.content) > 80
                        else item.content
                    )

                    table.add_row(
                        item.key, item.topic, str(item.usage_count), last_used, preview
                    )

                console.print(table)
            else:
                console.print(f"✗ No results found for: '{query}'", style="yellow")

        except Exception as e:
            console.print(f"✗ Error searching knowledge: {e}", style="red")

    @memory_app.command("stats")
    def memory_stats():
        """Show memory system statistics."""
        try:
            manager = MemoryManager()

            stats = manager.get_memory_stats()

            # Overview table
            overview_table = Table(title="Memory System Overview")
            overview_table.add_column("Metric", style="cyan")
            overview_table.add_column("Value", style="green")

            overview_table.add_row("Total Items", str(stats.get("total_items", 0)))
            overview_table.add_row("Total Topics", str(stats.get("total_topics", 0)))
            overview_table.add_row(
                "Total Favorites", str(stats.get("total_favorites", 0))
            )
            overview_table.add_row(
                "Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}"
            )

            console.print(overview_table)

            # Topic distribution
            topics = stats.get("topics", {})
            if topics:
                topic_table = Table(title="Topics")
                topic_table.add_column("Topic", style="magenta")
                topic_table.add_column("Items", style="green")

                for topic, count in sorted(topics.items()):
                    topic_table.add_row(topic, str(count))

                console.print(topic_table)

        except Exception as e:
            console.print(f"✗ Error getting stats: {e}", style="red")

except ImportError:
    memory_app = None

console = Console()


@app.command()
def chat(message: str):
    """Send a single prompt to the model and print the reply."""
    cfg = get_config()

    async def _run():
        try:
            reply = await asyncio.wait_for(
                query_llm(message, stream=False), timeout=get_config().llm_timeout_sec
            )
        except asyncio.TimeoutError:
            console.print("[timeout: LLM did not respond in time]", style="yellow")
            return
        except LLMBackendError:
            console.print(
                "LLM backend unavailable. Is Ollama running on 127.0.0.1:11434?",
                style="yellow",
            )
            return
        if isinstance(reply, str):
            log_conversation(message, reply, source="cli")
            console.print(reply)
        else:
            console.print("Unexpected response type", style="red")

    asyncio.run(_run())


@app.command()
def repl():
    """Interactive REPL with ephemeral memory (current session only)."""
    cfg = get_config()
    console.print("Vega2.0 REPL. Type /exit to quit.")
    history: list[tuple[str, str]] = []

    async def _ask(prompt: str) -> str:
        try:
            resp = await asyncio.wait_for(
                query_llm(prompt, stream=False), timeout=get_config().llm_timeout_sec
            )
        except asyncio.TimeoutError:
            return "[timeout: LLM did not respond in time]"
        except LLMBackendError:
            return "[LLM backend unavailable. Start Ollama on 127.0.0.1:11434]"
        except Exception as exc:
            return f"[error: {str(exc)[:200]}]"
        return str(resp)

    while True:
        try:
            prompt = console.input("[bold cyan]> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye")
            break
        if prompt.strip() in {"/exit", ":q", "quit"}:
            break
        reply = asyncio.run(_ask(prompt))
        history.append((prompt, reply))
        log_conversation(prompt, reply, source="cli")
        console.print(f"[bold green]{reply}")


@app.command(name="history")
def history_cmd(limit: int = typer.Option(20, help="Number of rows to show")):
    """Show last N conversations from SQLite."""
    rows = get_history(limit=limit)
    table = Table(title=f"Last {limit} conversations")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("source")
    table.add_column("prompt")
    table.add_column("response")
    for r in rows:
        table.add_row(
            str(r["id"]), r["ts"], r["source"], r["prompt"][:60], r["response"][:60]
        )
    console.print(table)


@integrations_app.command(name="test")
def integrations_test_cmd():
    """Send a Slack test message using configured webhook."""
    cfg = get_config()
    ok = send_slack_message(cfg.slack_webhook_url, "Vega2.0 integration test message")
    if ok:
        console.print("Slack message sent")
    else:
        console.print("Slack webhook not configured or failed", style="yellow")


@search_app.command("web")
def search_web_cli(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Web search via DuckDuckGo (privacy-friendly)."""
    from integrations.search import web_search

    items = web_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    table = Table(title=f"Web results for: {query}")
    table.add_column("title")
    table.add_column("url")
    table.add_column("snippet")
    for it in items:
        table.add_row(
            it.get("title") or "", it.get("href") or "", (it.get("snippet") or "")[:100]
        )
    console.print(table)


@search_app.command("images")
def search_images_cli(
    query: str = typer.Argument(..., help="Image query"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Image search via DuckDuckGo."""
    from integrations.search import image_search

    items = image_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    table = Table(title=f"Image results for: {query}")
    table.add_column("title")
    table.add_column("image")
    table.add_column("thumbnail")
    for it in items:
        table.add_row(
            it.get("title") or "", it.get("image") or "", it.get("thumbnail") or ""
        )
    console.print(table)


@search_app.command("research")
def search_research_cli(
    query: str = typer.Argument(..., help="Research topic"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Web research: fetch results and summarize via LLM."""
    from integrations.search import web_search
    from llm import query_llm, LLMBackendError

    items = web_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    ctx = []
    for i, it in enumerate(items, start=1):
        ctx.append(
            f"[{i}] {it.get('title') or ''} - {it.get('href') or ''}\n{it.get('snippet') or ''}"
        )
    prompt = (
        "Summarize the following search results into 5-8 bullet points with 2-3 top links at end.\n\n"
        + "\n\n".join(ctx)
    )
    try:
        result = asyncio.run(
            asyncio.wait_for(
                query_llm(prompt, stream=False), timeout=get_config().llm_timeout_sec
            )
        )
        summary = result if isinstance(result, str) else "[unexpected summary type]"
    except Exception as exc:
        summary = f"[error summarizing: {str(exc)[:200]}]"
    console.print("\n[bold]Summary[/bold]\n" + str(summary))
    table = Table(title=f"Top results for: {query}")
    table.add_column("title")
    table.add_column("url")
    for it in items[:5]:
        table.add_row(it.get("title") or "", it.get("href") or "")
    console.print(table)


@db_app.command("backup")
def db_backup(
    out: str = typer.Option(None, help="Output path; defaults to vega.db.backup")
):
    from db import backup_db

    path = backup_db(out)
    console.print(f"Backup written to {path}")


@db_app.command("vacuum")
def db_vacuum():
    from db import vacuum_db

    vacuum_db()
    console.print("VACUUM completed")


@db_app.command("export")
def db_export(
    path: str = typer.Argument("conversations.jsonl"),
    limit: Optional[int] = typer.Option(None),
):
    from db import export_jsonl

    p = export_jsonl(path, limit=limit)
    console.print(f"Exported to {p}")


@db_app.command("import")
def db_import(path: str = typer.Argument(...)):
    from db import import_jsonl

    n = import_jsonl(path)
    console.print(f"Imported {n} rows")


@db_app.command("purge")
def db_purge(days: int = typer.Argument(..., help="Delete rows older than N days")):
    from db import purge_old

    n = purge_old(days)
    console.print(f"Purged {n} rows older than {days} days")


@db_app.command("search")
def db_search(q: str = typer.Argument(...), limit: int = typer.Option(20)):
    from db import search_conversations

    rows = search_conversations(q, limit=limit)
    table = Table(title=f"Search '{q}' ({len(rows)})")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("source")
    table.add_column("prompt")
    table.add_column("response")
    for r in rows:
        table.add_row(
            str(r["id"]), r["ts"], r["source"], r["prompt"][:60], r["response"][:60]
        )
    console.print(table)


@gen_app.command("show")
def gen_show():
    from llm import get_generation_settings

    s = get_generation_settings()
    table = Table(title="Generation Settings")
    table.add_column("key")
    table.add_column("value")
    for k, v in s.items():
        table.add_row(str(k), str(v))
    console.print(table)


@gen_app.command("set")
def gen_set(
    temperature: Optional[float] = typer.Option(None),
    top_p: Optional[float] = typer.Option(None),
    top_k: Optional[int] = typer.Option(None),
    repeat_penalty: Optional[float] = typer.Option(None),
    presence_penalty: Optional[float] = typer.Option(None),
    frequency_penalty: Optional[float] = typer.Option(None),
):
    from llm import set_generation_settings, get_generation_settings

    set_generation_settings(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    gen_show()


@gen_app.command("dynamic")
def gen_dynamic(enable: bool = typer.Argument(True)):
    from llm import set_generation_settings

    set_generation_settings(dynamic_generation=bool(enable))
    gen_show()


@gen_app.command("reset")
def gen_reset():
    from llm import reset_generation_settings

    reset_generation_settings()
    gen_show()


@dataset_app.command(name="build")
def dataset_build(path: str = typer.Argument(".", help="Input directory with files")):
    """Create datasets/output.jsonl from the given directory."""
    from datasets.prepare_dataset import build_dataset

    out = build_dataset(path)
    console.print(f"Dataset written to {out}")


@app.command()
def train(
    config: str = typer.Option("training/config.yaml", help="Training config path")
):
    """Run fine-tuning pipeline using Hugging Face + Accelerate."""
    from training.train import run_training

    run_training(config)


@app.command()
def feedback(
    conversation_id: int = typer.Argument(..., help="Conversation ID to annotate"),
    rating: Optional[int] = typer.Option(None, help="1-5 rating"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    notes: Optional[str] = typer.Option(None, help="Freeform notes"),
    reviewed: Optional[bool] = typer.Option(None, help="Mark as reviewed (true/false)"),
):
    """Attach feedback/metadata to a conversation row for later curation."""
    ok = set_feedback(
        conversation_id, rating=rating, tags=tags, notes=notes, reviewed=reviewed
    )
    if ok:
        console.print("Feedback saved")
    else:
        console.print("Conversation not found", style="red")


@learn_app.command("curate")
def learn_curate(
    min_rating: int = typer.Option(4, help="Minimum rating to include"),
    reviewed_only: bool = typer.Option(False, help="Only include reviewed rows"),
    out_path: str = typer.Option("datasets/curated.jsonl", help="Output JSONL path"),
):
    from learning.learn import curate_dataset

    path = curate_dataset(
        min_rating=min_rating, reviewed_only=reviewed_only, out_path=out_path
    )
    console.print(f"Curated dataset written to {path}")


@learn_app.command("evaluate")
def learn_evaluate(
    eval_file: str = typer.Argument(..., help="Eval JSONL with {prompt,response}"),
    system_prompt: Optional[str] = typer.Option(
        None, help="Inline system prompt override"
    ),
):
    import asyncio
    from learning.learn import evaluate_model

    score = asyncio.run(evaluate_model(eval_file, system_prompt=system_prompt))
    console.print(f"Average score: {score:.3f}")


@learn_app.command("optimize-prompt")
def learn_optimize_prompt(
    candidates_file: str = typer.Argument(
        ..., help="Text file with one candidate per line"
    ),
    eval_file: str = typer.Argument(..., help="Eval JSONL file with {prompt,response}"),
    out_dir: str = typer.Option("prompts", help="Directory to write system_prompt.txt"),
):
    import asyncio
    from learning.learn import optimize_system_prompt

    scores = asyncio.run(
        optimize_system_prompt(candidates_file, eval_file, out_dir=out_dir)
    )
    # Pretty print top 3
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    table = Table(title="Top system prompts")
    table.add_column("score")
    table.add_column("prompt")
    for p, s in top:
        table.add_row(f"{s:.3f}", p[:80])
    console.print(table)


if __name__ == "__main__":
    app()  # Typer entry


# OSINT CLI
@osint_app.command("dns")
def osint_dns(hostname: str = typer.Argument(...)):
    from integrations.osint import dns_lookup

    res = dns_lookup(hostname)
    table = Table(title=f"DNS {hostname}")
    table.add_column("addresses")
    table.add_row(", ".join(res.addresses))
    console.print(table)


@osint_app.command("rdns")
def osint_rdns(ip: str = typer.Argument(...)):
    from integrations.osint import reverse_dns

    names = reverse_dns(ip)
    table = Table(title=f"rDNS {ip}")
    table.add_column("names")
    table.add_row(", ".join(names) or "<none>")
    console.print(table)


@osint_app.command("headers")
def osint_headers(url: str = typer.Argument(...)):
    from integrations.osint import http_headers

    res = http_headers(url)
    table = Table(title=f"Headers {url}")
    table.add_column("status")
    table.add_column("header")
    table.add_column("value")
    first = True
    for k, v in res.headers.items():
        if first:
            table.add_row(str(res.status), k, v)
            first = False
        else:
            table.add_row("", k, v)
    console.print(table)


@osint_app.command("ssl")
def osint_ssl(host: str = typer.Argument(...), port: int = typer.Option(443)):
    from integrations.osint import ssl_cert_info

    info = ssl_cert_info(host, port=port)
    if not info:
        console.print("No certificate info", style="yellow")
        return
    table = Table(title=f"SSL {host}:{port}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("subject", info.subject)
    table.add_row("issuer", info.issuer)
    table.add_row("not_before", info.not_before)
    table.add_row("not_after", info.not_after)
    console.print(table)


@osint_app.command("robots")
def osint_robots(url: str = typer.Argument(...)):
    from integrations.osint import robots_txt

    txt = robots_txt(url)
    console.print(txt or "<empty>")


@osint_app.command("whois")
def osint_whois(domain: str = typer.Argument(...)):
    from integrations.osint import whois_lookup

    data = whois_lookup(domain)
    if "error" in data:
        console.print(f"Error: {data['error']}", style="yellow")
        return
    table = Table(title=f"WHOIS {domain}")
    table.add_column("key")
    table.add_column("value")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(table)


@osint_app.command("username")
def osint_username(
    username: str = typer.Argument(...),
    include_nsfw: bool = typer.Option(False),
    sites: Optional[str] = typer.Option(
        None, help="Comma-separated site filter (e.g., github,reddit)"
    ),
):
    from integrations.osint import username_search

    site_list = [s.strip() for s in sites.split(",")] if sites else None
    items = username_search(username, include_nsfw=include_nsfw, sites=site_list)
    table = Table(title=f"Username search: {username}")
    table.add_column("site")
    table.add_column("exists")
    table.add_column("status")
    table.add_column("url")
    for it in items:
        table.add_row(
            it["site"], "yes" if it["exists"] else "no", str(it["status"]), it["url"]
        )
    console.print(table)


# Net CLI
def _parse_ports(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            for p in range(start, end + 1):
                if 1 <= p <= 65535:
                    out.append(p)
        else:
            try:
                p = int(part)
            except Exception:
                continue
            if 1 <= p <= 65535:
                out.append(p)
    # de-dupe and cap
    out = sorted(set(out))[:1024]
    return out


@net_app.command("scan")
def net_scan(host: str = typer.Argument(...), ports: str = typer.Argument(...)):
    from integrations.osint import tcp_scan

    port_list = _parse_ports(ports)
    if not port_list:
        console.print("No valid ports", style="yellow")
        return

    async def _run():
        res = await tcp_scan(host, port_list)
        table = Table(title=f"Scan {host}")
        table.add_column("port")
        table.add_column("state")
        for p, s in res:
            table.add_row(str(p), s)
        console.print(table)

    asyncio.run(_run())
