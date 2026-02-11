import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set env before any app import
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-api-tests")


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


def _make_response(content_blocks, stop_reason="end_turn"):
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    return resp


def _make_stream_ctx(chunks: list[str]):
    mock_stream = AsyncMock()

    async def text_iter():
        for c in chunks:
            yield c

    mock_stream.text_stream = text_iter()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    return mock_stream


@pytest.fixture()
def mock_anthropic():
    """Patch the AsyncAnthropic client used by the ChatService singleton in the route module."""
    # Import here so the env var is already set
    from app.routes import chat as chat_module
    from app.services.chat_service import _sessions

    _sessions.clear()

    # If _service was None because key was missing at first import, recreate it
    if chat_module._service is None:
        with patch("app.services.chat_service.anthropic.AsyncAnthropic"):
            from app.services.chat_service import ChatService
            chat_module._service = ChatService()

    mock_client = MagicMock()
    chat_module._service.client = mock_client
    yield mock_client

    _sessions.clear()


@pytest.fixture()
def client(mock_anthropic):
    """httpx AsyncClient bound to the FastAPI app."""
    import httpx
    from app.main import app
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# -- Helper factories exposed as fixtures --

@pytest.fixture()
def make_text_block():
    return _make_text_block


@pytest.fixture()
def make_tool_use_block():
    return _make_tool_use_block


@pytest.fixture()
def make_response():
    return _make_response


@pytest.fixture()
def make_stream_ctx():
    return _make_stream_ctx


def setup_normal_chat(mock_client, text_chunks: list[str]):
    """Configure mock for a normal (no tool use) chat response."""
    mock_client.messages.create = AsyncMock(
        return_value=_make_response([_make_text_block("".join(text_chunks))], "end_turn")
    )
    mock_client.messages.stream = MagicMock(
        return_value=_make_stream_ctx(text_chunks)
    )


def setup_tool_use_chat(mock_client, tool_id, tool_name, tool_input, final_chunks):
    """Configure mock for a tool_use -> end_turn flow."""
    tool_response = _make_response(
        [
            _make_text_block(""),
            _make_tool_use_block(tool_id, tool_name, tool_input),
        ],
        stop_reason="tool_use",
    )
    final_response = _make_response([_make_text_block("".join(final_chunks))], "end_turn")

    mock_client.messages.create = AsyncMock(side_effect=[tool_response, final_response])
    mock_client.messages.stream = MagicMock(return_value=_make_stream_ctx(final_chunks))
