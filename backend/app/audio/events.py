"""Server-Sent Events (SSE) event bus for conversation updates.

Pushes transcription segments, speaker labels, and voice-to-face links
to the frontend as they are produced by the audio pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("forgetmenot.audio.events")


class ConversationEventBus:
    """Async fan-out bus for conversation events over SSE."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._lock = asyncio.Lock()
        logger.info("Conversation event bus initialized")

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers of a session."""
        payload = json.dumps(event, separators=(",", ":"))
        async with self._lock:
            targets = list(self._subscribers.get(session_id, set()))
        if not targets:
            logger.debug("No subscribers for session %s; dropping event", session_id)
            return
        for queue in targets:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("SSE queue full for session %s, dropping event", session_id)

    async def subscribe(self, session_id: str) -> asyncio.Queue[str]:
        """Register a new subscriber queue for a session."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        async with self._lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = set()
            self._subscribers[session_id].add(queue)
        logger.info("New SSE subscriber for session %s (total=%d)", session_id, len(self._subscribers.get(session_id, set())))
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            subscribers = self._subscribers.get(session_id)
            if subscribers:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[session_id]

    async def stream(self, session_id: str):
        """Async generator yielding events for a subscriber."""
        queue = await self.subscribe(session_id)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield payload
                except asyncio.TimeoutError:
                    # Send a keep-alive comment.
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await self.unsubscribe(session_id, queue)

    def session_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, set()))
