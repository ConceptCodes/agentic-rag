from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ContextTracker:
    """Track chunks read during one A-RAG retrieval episode."""

    read_chunk_ids: set[str] = field(default_factory=set)

    def split_unread(self, chunk_ids: list[str]) -> tuple[list[str], list[str]]:
        """Return unread chunk IDs and repeated chunk IDs in request order."""
        unread: list[str] = []
        repeated: list[str] = []
        for chunk_id in chunk_ids:
            if chunk_id in self.read_chunk_ids:
                repeated.append(chunk_id)
            else:
                unread.append(chunk_id)
        return unread, repeated

    def mark_read(self, chunk_ids: list[str]) -> None:
        """Record chunk IDs as read."""
        self.read_chunk_ids.update(chunk_ids)
