import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.chat import ChatMessage, ChatRequest


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        yield


@pytest.fixture
def chat_service(mock_env):
    with patch("app.services.chat_service.anthropic.AsyncAnthropic"):
        from app.services.chat_service import ChatService, _sessions
        _sessions.clear()
        return ChatService()


def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id, name, tool_input):
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


def _make_stream_ctx(chunks):
    mock_stream = AsyncMock()

    async def text_iter():
        for c in chunks:
            yield c

    mock_stream.text_stream = text_iter()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    return mock_stream


# --- No tool use (direct streaming) ---

@pytest.mark.asyncio
async def test_chat_stream_no_tool_use(chat_service):
    # First call: non-streaming returns end_turn with text
    chat_service.client.messages.create = AsyncMock(
        return_value=_make_response([_make_text_block("你好")], "end_turn")
    )
    # Then streaming final response
    chat_service.client.messages.stream = MagicMock(
        return_value=_make_stream_ctx(["你", "好"])
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

    # Round 1: Claude wants to call get_weather
    tool_response = _make_response(
        [
            _make_text_block("让我查一下天气"),
            _make_tool_use_block("call_1", "get_weather", {"city": "北京"}),
        ],
        stop_reason="tool_use",
    )
    # Round 2: after tool result, Claude returns end_turn
    final_response = _make_response([_make_text_block("北京今天22°C")], "end_turn")

    chat_service.client.messages.create = AsyncMock(
        side_effect=[tool_response, final_response]
    )
    chat_service.client.messages.stream = MagicMock(
        return_value=_make_stream_ctx(["北京", "今天", "晴"])
    )

    chunks = []
    async for chunk in chat_service.chat_stream("s2", "北京天气怎么样"):
        chunks.append(chunk)

    # 3 text deltas + 1 message_stop
    assert len(chunks) == 4

    # Verify history includes tool use loop messages
    history = _sessions["s2"]
    # user -> assistant(tool_use) -> user(tool_result) -> assistant(final)
    assert history[0] == {"role": "user", "content": "北京天气怎么样"}
    assert history[1]["role"] == "assistant"
    # assistant content should have text + tool_use blocks
    assert any(b["type"] == "tool_use" for b in history[1]["content"])
    # tool_result
    assert history[2]["role"] == "user"
    assert history[2]["content"][0]["type"] == "tool_result"
    assert history[2]["content"][0]["tool_use_id"] == "call_1"
    # final assistant text
    assert history[3] == {"role": "assistant", "content": "北京今天晴"}


# --- Session history ---

@pytest.mark.asyncio
async def test_session_history_maintained(chat_service):
    from app.services.chat_service import _sessions

    chat_service.client.messages.create = AsyncMock(
        return_value=_make_response([_make_text_block("Hello!")], "end_turn")
    )
    chat_service.client.messages.stream = MagicMock(
        return_value=_make_stream_ctx(["Hello!"])
    )

    async for _ in chat_service.chat_stream("sess-1", "Hi"):
        pass

    assert len(_sessions["sess-1"]) == 2
    assert _sessions["sess-1"][0] == {"role": "user", "content": "Hi"}
    assert _sessions["sess-1"][1] == {"role": "assistant", "content": "Hello!"}


# --- Basic model tests ---

def test_chat_service_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
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
