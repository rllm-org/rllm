"""Thread-safe session pool for tracking ARL sandbox sessions.

Allows reconnecting to existing sessions after SWEEnv re-creation,
tracking active sessions for cleanup, and looking up session IDs
for re-attachment via SandboxSession.attach().
"""

import threading
from dataclasses import dataclass


@dataclass
class SessionEntry:
    session_id: str
    pool_ref: str
    gateway_url: str
    namespace: str


class SessionPool:
    """Thread-safe registry of active ARL sessions."""

    _instance = None
    _class_lock = threading.Lock()

    def __init__(self):
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SessionPool":
        """Get or create the singleton SessionPool instance."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, key: str, entry: SessionEntry):
        """Register a session entry for a given key (e.g., task instance_id)."""
        with self._lock:
            self._sessions[key] = entry

    def get(self, key: str) -> SessionEntry | None:
        """Look up a session entry by key."""
        with self._lock:
            return self._sessions.get(key)

    def remove(self, key: str):
        """Remove a session entry."""
        with self._lock:
            self._sessions.pop(key, None)

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._sessions.clear()

    def keys(self) -> list[str]:
        """Return all registered keys."""
        with self._lock:
            return list(self._sessions.keys())
