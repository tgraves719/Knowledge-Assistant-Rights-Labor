from backend.generation.context import bind_session_context, get_session_context


def test_session_context_bind_without_db():
    bind_session_context(
        "session-test",
        union_local_id="local-7",
        union_id="union-7",
        user_id="user-7",
        message_retention_enabled=False,
    )
    ctx = get_session_context("session-test")
    assert ctx.session_id == "session-test"

