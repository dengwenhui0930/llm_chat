import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.chat import ChatMessage, ChatRequest


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        yield


@pytest.fixture
def chat_service(mock_env):
    with patch("app.services.chat_service.AsyncOpenAI"):
        from app.services.chat_service import ChatService, _sessions
        _sessions.clear()
        return ChatService()


def _make_choice(content=None, tool_calls=None):
    message = MagicMock()
    message.content = content or ""
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    return choice


def _make_tool_call(call_id, name, arguments_dict):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict, ensure_ascii=False)
    return tc


def _make_response(content=None, tool_calls=None):
    resp = MagicMock()
    resp.choices = [_make_choice(content, tool_calls)]
    return resp


async def _make_stream_chunks(chunks):
    for text in chunks:
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
        yield chunk


# --- No tool use (direct streaming) ---

@pytest.mark.asyncio
async def test_chat_stream_no_tool_use(chat_service):
    chat_service.client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_response(content="你好"),
            _make_stream_chunks(["你", "好"]),
        ]
    )

    chunks = []
    async for chunk in chat_service.chat_stream("s1", "Hi"):
        chunks.append(chunk)

    assert len(chunks) == 3
    d1 = json.loads(chunks[0].removeprefix("data: ").strip())
    assert d1 == {"type": "content_block_delta", "text": "你"}
    d2 = json.loads(chunks[1].removeprefix("data: ").strip())
    assert d2 == {"type": "content_block_delta", "text": "好"}
    stop = json.loads(chunks[2].removeprefix("data: ").strip())
    assert stop == {"type": "message_stop"}


# --- Tool use flow ---

@pytest.mark.asyncio
async def test_chat_stream_with_tool_use(chat_service):
    from app.services.chat_service import _sessions

    tool_response = _make_response(
        content="",
        tool_calls=[_make_tool_call("call_1", "get_weather", {"city": "北京"})],
    )
    final_response = _make_response(content="北京今天22°C")

    chat_service.client.chat.completions.create = AsyncMock(
        side_effect=[
            tool_response,
            final_response,
            _make_stream_chunks(["北京", "今天", "晴"]),
        ]
    )

    chunks = []
    async for chunk in chat_service.chat_stream("s2", "北京天气怎么样"):
        chunks.append(chunk)

    assert len(chunks) == 4

    history = _sessions["s2"]
    assert history[0] == {"role": "user", "content": "北京天气怎么样"}
    # assistant with tool_calls
    assert history[1]["role"] == "assistant"
    assert "tool_calls" in history[1]
    assert history[1]["tool_calls"][0]["id"] == "call_1"
    # tool result
    assert history[2]["role"] == "tool"
    assert history[2]["tool_call_id"] == "call_1"
    # final assistant
    assert history[3] == {"role": "assistant", "content": "北京今天晴"}


# --- Session history ---

@pytest.mark.asyncio
async def test_session_history_maintained(chat_service):
    from app.services.chat_service import _sessions

    chat_service.client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_response(content="Hello!"),
            _make_stream_chunks(["Hello!"]),
        ]
    )

    async for _ in chat_service.chat_stream("sess-1", "Hi"):
        pass

    assert len(_sessions["sess-1"]) == 2
    assert _sessions["sess-1"][0] == {"role": "user", "content": "Hi"}
    assert _sessions["sess-1"][1] == {"role": "assistant", "content": "Hello!"}


# --- Basic model tests ---

def test_chat_service_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            from app.services.chat_service import ChatService
            ChatService()


def test_chat_message_model():
    msg = ChatMessage(role="user", content="test")
    assert msg.role == "user"
    assert msg.content == "test"


def test_chat_request_model():
    req = ChatRequest(session_id="abc", message="hello")
    assert req.session_id == "abc"
    assert req.message == "hello"
