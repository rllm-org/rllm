from abc import ABC, abstractmethod
from typing import Any


def extract_searchable_text(episode_data: dict) -> str:
    """Extract searchable text from episode data.

    Extracts text from task, observations, actions, and model responses
    to create a searchable string for full-text search indexing.
    """
    parts = []

    # Task question/description
    task = episode_data.get("task", {})
    if isinstance(task, dict):
        parts.append(str(task.get("question", "")))
        parts.append(str(task.get("description", "")))
    elif isinstance(task, str):
        parts.append(task)

    # Trajectory steps
    for traj in episode_data.get("trajectories", []):
        for step in traj.get("steps", []):
            parts.extend(
                [
                    str(step.get("observation", "")),
                    str(step.get("action", "")),
                    str(step.get("model_response", "")),
                ]
            )
            # Also extract from chat_completions if present
            for msg in step.get("chat_completions", []) or []:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))

    return " ".join(filter(None, parts))


class DataStore(ABC):
    """Abstract base class for data storage implementations."""

    @abstractmethod
    def init_db(self):
        """Initialize the database schema."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the data store (destructive, mainly for tests)."""
        pass

    @abstractmethod
    def create_session(self, project: str, experiment: str, config: dict[str, Any], source_metadata: dict[str, Any]) -> str:
        """Create a new training session."""
        pass

    @abstractmethod
    def log_metrics(self, session_id: str, step: int, data: dict[str, Any]) -> dict[str, Any] | None:
        """Log metrics for a session."""
        pass

    @abstractmethod
    def append_episode(self, session_id: str, episode_data: dict[str, Any]):
        """Append an episode to a session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session details."""
        pass

    @abstractmethod
    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Retrieve all sessions."""
        pass

    @abstractmethod
    def complete_session(self, session_id: str) -> dict[str, Any] | None:
        """Mark a session as completed."""
        pass

    @abstractmethod
    def get_new_metrics(self, session_id: str, last_id: int) -> list[dict[str, Any]]:
        """Retrieve metrics since last_id."""
        pass

    @abstractmethod
    def get_metrics(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve metrics for a session."""
        pass

    @abstractmethod
    def get_episodes(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve episodes for a session."""
        pass

    @abstractmethod
    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """Retrieve a specific episode."""
        pass

    @abstractmethod
    def search_episodes(self, query: str, session_id: str | None = None, limit: int = 50, step: int | None = None) -> dict[str, Any]:
        """Search episodes by text content.

        Args:
            query: Search query string
            session_id: Optional session ID to filter results
            limit: Maximum number of results to return
            step: Optional step number to filter results

        Returns:
            Dict with:
                - episodes: List of matching episodes (PostgreSQL includes 'rank' field)
                - matched_terms: List of terms used for matching (stemmed for PostgreSQL)
        """
        pass
