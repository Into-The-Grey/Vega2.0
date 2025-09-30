"""Lightweight bridge for IDE live visibility."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from .monitoring import LOG_DIR


class LiveViewServer:
    """Simple TCP broadcast server emitting newline-delimited JSON updates."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None
        self._clients: Set[asyncio.StreamWriter] = set()
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_client, self.host, self.port)
        asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for client in list(self._clients):
            client.close()
            await client.wait_closed()

    async def publish(self, event: Dict[str, Any]) -> None:
        await self._queue.put(event)

    async def _broadcast_loop(self) -> None:
        while True:
            event = await self._queue.get()
            payload = json.dumps(event) + "\n"
            to_remove: List[asyncio.StreamWriter] = []
            for writer in list(self._clients):
                try:
                    writer.write(payload.encode("utf-8"))
                    await writer.drain()
                except Exception:
                    to_remove.append(writer)
            for writer in to_remove:
                self._clients.discard(writer)

    async def _on_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._clients.add(writer)
        try:
            while not reader.at_eof():
                await reader.readline()
        finally:
            self._clients.discard(writer)
            writer.close()
            await writer.wait_closed()


class IDEActivityLogger:
    """Writes textual narrative suitable for VS Code output panels."""

    def __init__(self) -> None:
        self.logfile = LOG_DIR / "ide_activity.log"

    def log(self, message: str, **details: Any) -> None:
        entry = {
            "message": message,
            "details": details,
            "pid": os.getpid(),
        }
        with self.logfile.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
