import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set env before any app import
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-api-tests")


def _make_choice(content=None, tool_calls=None):
    """Build a mock Choice object (OpenAI format)."""
    message = MagicMock()
    message.content = content or ""
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    return choice


def _make_tool_call(call_id, name, arguments_dict):
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict, ensure_ascii=False)
    return tc


def _make_response(content=None, tool_calls=None):
    """Build a mock ChatCompletion response."""
    resp = MagicMock()
    resp.choices = [_make_choice(content, tool_calls)]
    return resp


def _make_stream_chunks(text_chunks: list[str]):
    """Build an async iterable of stream chunks (OpenAI format)."""
    async def stream_iter():
        for text in text_chunks:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = text
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk
    return stream_iter()


@pytest.fixture()
def mock_openai():
    """Patch the OpenAI client used by the ChatService singleton."""
    from app.routes import chat as chat_module
    from app.services.chat_service import _sessions

    _sessions.clear()

    if chat_module._service is None:
        with patch("app.services.chat_service.AsyncOpenAI"):
            from app.services.chat_service import ChatService
            chat_module._service = ChatService()

    mock_client = MagicMock()
    chat_module._service.client = mock_client
    yield mock_client

    _sessions.clear()


@pytest.fixture()
def client(mock_openai):
    """httpx AsyncClient bound to the FastAPI app."""
    import httpx
    from app.main import app
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# -- Helper factories exposed as fixtures --

@pytest.fixture()
def make_response():
    return _make_response


@pytest.fixture()
def make_tool_call():
    return _make_tool_call


@pytest.fixture()
def make_stream_chunks():
    return _make_stream_chunks


def setup_normal_chat(mock_client, text_chunks: list[str]):
    """Configure mock for a normal (no tool use) chat response."""
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_response(content="".join(text_chunks)),  # non-streaming
            _make_stream_chunks(text_chunks),               # streaming
        ]
    )


def setup_tool_use_chat(mock_client, tool_id, tool_name, tool_input, final_chunks):
    """Configure mock for a tool_use -> final text flow."""
    tool_response = _make_response(
        content="",
        tool_calls=[_make_tool_call(tool_id, tool_name, tool_input)],
    )
    final_response = _make_response(content="".join(final_chunks))

    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            tool_response,      # round 1: tool call
            final_response,     # round 2: no more tools
            _make_stream_chunks(final_chunks),  # streaming final
        ]
    )
