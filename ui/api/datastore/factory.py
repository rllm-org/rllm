import os

from .base import DataStore
from .sqlite_store import SQLiteStore


def get_datastore(url: str | None = None) -> DataStore:
    """
    Factory to create a DataStore instance based on the connection URL.

    URL resolution order:
    1. Explicit url parameter
    2. DATABASE_URL environment variable
    3. Default to SQLite (rllm_ui.db)

    Supported URL schemes:
    - sqlite:// or .db file path -> SQLiteStore
    - postgresql:// or postgres:// -> PostgresStore
    """
    # Check environment variable if no URL provided
    if url is None:
        url = os.environ.get("DATABASE_URL")

    # Default to SQLite if still no URL
    if not url or url.startswith("sqlite") or url.endswith(".db"):
        return SQLiteStore()
    elif url.startswith("postgresql") or url.startswith("postgres"):
        from .postgres_store import PostgresStore

        return PostgresStore(url)
    else:
        raise NotImplementedError(f"DataStore for {url} not implemented yet")
