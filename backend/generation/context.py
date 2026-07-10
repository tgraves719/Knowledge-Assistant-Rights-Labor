"""
Conversation Context Manager

Maintains conversation history for multi-turn interactions.
Enables follow-up questions like "what about them?" to maintain context.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timedelta

from backend.platform.service_container import ServiceContainer, build_service_container


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    detected_entities: dict[str, Any] = field(default_factory=dict)
    retrieval_context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def summary(self, max_length: int = 200) -> str:
        """Get a condensed version for context injection."""
        answer_preview = self.answer[:max_length]
        if len(self.answer) > max_length:
            answer_preview += "..."
        return f"Q: {self.question}\nA: {answer_preview}"


class ConversationContext:
    """
    Manages conversation history for a user session.
    
    Designed to be:
    - Contract-agnostic (works across any contract)
    - Memory-efficient (caps history length)
    - Context-aware (provides relevant history to LLM)
    """
    
    def __init__(
        self, 
        max_turns: int = 5,
        session_timeout_minutes: int = 30,
        session_id: Optional[str] = None,
    ):
        """
        Initialize conversation context.
        
        Args:
            max_turns: Maximum conversation turns to retain
            session_timeout_minutes: Clear history after this many minutes of inactivity
        """
        self.max_turns = max_turns
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.session_id = session_id or ""
        self.history: list[ConversationTurn] = []
        self.last_activity: datetime = datetime.now()
        
        # Track entities mentioned for pronoun resolution
        self.mentioned_entities: dict[str, str] = {}
        # e.g., {"classification": "courtesy clerk", "topic": "health insurance"}
    
    def _check_timeout(self):
        """Clear history if session has timed out."""
        if datetime.now() - self.last_activity > self.session_timeout:
            self.clear()
    
    def add_turn(
        self, 
        question: str, 
        answer: str, 
        citations: list[str] = None,
        detected_entities: dict[str, Any] = None,
        retrieval_context: dict[str, Any] = None,
    ):
        """
        Add a conversation turn to history.
        
        Args:
            question: User's question
            answer: Karl's response
            citations: List of citations used
            detected_entities: Entities detected in this turn
        """
        self._check_timeout()
        self.last_activity = datetime.now()
        
        turn = ConversationTurn(
            question=question,
            answer=answer,
            citations=citations or [],
            detected_entities=dict(detected_entities or {}),
            retrieval_context=dict(retrieval_context or {}),
        )
        
        self.history.append(turn)
        
        # Update mentioned entities
        if detected_entities:
            self.mentioned_entities.update(detected_entities)
        
        # Trim to max turns
        while len(self.history) > self.max_turns:
            self.history.pop(0)
    
    def get_recent_context(self, n_turns: int = 3) -> str:
        """
        Get formatted context from recent turns.
        
        Returns a string suitable for injection into the system prompt.
        """
        self._check_timeout()
        
        if not self.history:
            return ""
        
        recent = self.history[-n_turns:]
        
        context_parts = ["## CONVERSATION HISTORY (for context on follow-up questions):"]
        for i, turn in enumerate(recent, 1):
            context_parts.append(f"\n### Turn {i}:")
            context_parts.append(turn.summary(max_length=300))
            if turn.citations:
                context_parts.append(f"Citations: {', '.join(turn.citations[:3])}")
        
        return "\n".join(context_parts)
    
    def get_entity_context(self) -> str:
        """
        Get context about mentioned entities for pronoun resolution.
        
        Helps resolve questions like "what about them?" or "how does that work?"
        """
        if not self.mentioned_entities:
            return ""
        
        parts = ["## PREVIOUSLY MENTIONED:"]
        for entity_type, value in self.mentioned_entities.items():
            parts.append(f"- {entity_type}: {value}")
        
        return "\n".join(parts)
    
    def get_full_context(self) -> str:
        """Get all context for prompt injection."""
        parts = []
        
        entity_ctx = self.get_entity_context()
        if entity_ctx:
            parts.append(entity_ctx)
        
        history_ctx = self.get_recent_context()
        if history_ctx:
            parts.append(history_ctx)
        
        if parts:
            parts.append("\n## CURRENT QUESTION:")
        
        return "\n\n".join(parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.mentioned_entities = {}
        self.last_activity = datetime.now()
    
    def get_last_topic(self) -> Optional[str]:
        """Get the topic from the last turn, if any."""
        return self.mentioned_entities.get("topic")
    
    def get_last_classification(self) -> Optional[str]:
        """Get the classification from the last turn, if any."""
        return self.mentioned_entities.get("classification")

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn."""
        self._check_timeout()
        if not self.history:
            return None
        return self.history[-1]

    def get_last_citations(self) -> list[str]:
        """Get citations from the most recent turn."""
        turn = self.get_last_turn()
        if not turn:
            return []
        return list(turn.citations or [])

    def get_last_retrieval_context(self) -> dict[str, Any]:
        """Get retrieval anchors from the most recent turn."""
        turn = self.get_last_turn()
        if not turn:
            return {}
        return dict(turn.retrieval_context or {})


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionManager:
    """
    Manages multiple conversation contexts by session ID.
    
    For production, this would be backed by Redis or similar.
    For now, in-memory with automatic cleanup.
    """
    
    def __init__(self, max_sessions: int = 1000):
        self.sessions: dict[str, ConversationContext] = {}
        self.max_sessions = max_sessions
        self._platform: ServiceContainer | None = None

    @property
    def platform(self) -> ServiceContainer:
        if self._platform is None:
            self._platform = build_service_container()
        return self._platform
    
    def get_context(self, session_id: str) -> ConversationContext:
        """Get or create a conversation context for a session."""
        if session_id not in self.sessions:
            # Cleanup old sessions if at capacity
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_old_sessions()
            
            ctx = ConversationContext(session_id=session_id)
            persisted = self.platform.chat_history.load_messages(session_id)
            for pair_index in range(0, len(persisted), 2):
                user_msg = persisted[pair_index]
                assistant_msg = persisted[pair_index + 1] if pair_index + 1 < len(persisted) else None
                if assistant_msg is None:
                    continue
                ctx.add_turn(
                    question=str(user_msg.get("content") or ""),
                    answer=str(assistant_msg.get("content") or ""),
                    citations=list((assistant_msg.get("metadata") or {}).get("citations") or []),
                    detected_entities=dict((assistant_msg.get("metadata") or {}).get("detected_entities") or {}),
                    retrieval_context=dict((assistant_msg.get("metadata") or {}).get("retrieval_context") or {}),
                )
            self.sessions[session_id] = ctx
        
        return self.sessions[session_id]

    def bind_session(
        self,
        session_id: str,
        *,
        union_local_id: str | None = None,
        union_id: str | None = None,
        user_id: str | None = None,
        message_retention_enabled: bool = False,
    ) -> None:
        self.platform.chat_history.bind_session(
            session_id=session_id,
            union_local_id=union_local_id,
            union_id=union_id,
            user_id=user_id,
            message_retention_enabled=message_retention_enabled,
        )

    def persist_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> None:
        self.platform.chat_history.persist_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            metadata=metadata,
        )
    
    def _cleanup_old_sessions(self):
        """Remove oldest sessions to make room."""
        # Sort by last activity and remove oldest 10%
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_activity
        )
        
        to_remove = len(sorted_sessions) // 10 or 1
        for session_id, _ in sorted_sessions[:to_remove]:
            del self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global session manager instance
_session_manager = SessionManager()


def get_session_context(session_id: str) -> ConversationContext:
    """Get conversation context for a session (convenience function)."""
    return _session_manager.get_context(session_id)


def clear_session_context(session_id: str):
    """Clear conversation context for a session."""
    _session_manager.clear_session(session_id)


def bind_session_context(
    session_id: str,
    *,
    union_local_id: str | None = None,
    union_id: str | None = None,
    user_id: str | None = None,
    message_retention_enabled: bool = False,
):
    _session_manager.bind_session(
        session_id,
        union_local_id=union_local_id,
        union_id=union_id,
        user_id=user_id,
        message_retention_enabled=message_retention_enabled,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test conversation context
    ctx = ConversationContext()
    
    # Simulate conversation
    ctx.add_turn(
        question="when do I get health insurance?",
        answer="As a Courtesy Clerk, you need 800 hours in an anniversary year...",
        citations=["Article 18, Section 54"],
        detected_entities={"topic": "health insurance", "classification": "courtesy clerk"}
    )
    
    ctx.add_turn(
        question="does it matter how long I've been here?",
        answer="Yes, length of service matters for benefits...",
        citations=["Article 40, Section 116"],
        detected_entities={"topic": "health insurance"}  # Same topic
    )
    
    print("=== Full Context ===")
    print(ctx.get_full_context())
    
    print("\n=== Last Topic ===")
    print(ctx.get_last_topic())

