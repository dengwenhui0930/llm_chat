import json
import io

import pytest

from tests.conftest import setup_normal_chat, setup_tool_use_chat


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE text into a list of JSON event payloads."""
    events = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ---------- 1. test_normal_chat ----------

@pytest.mark.asyncio
async def test_normal_chat(client, mock_anthropic):
    setup_normal_chat(mock_anthropic, ["你好", "，", "世界"])

    resp = await client.post("/chat", json={
        "session_id": "normal-1",
        "message": "你好",
    })

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    stop = [e for e in events if e["type"] == "message_stop"]

    assert len(deltas) == 3
    assert deltas[0]["text"] == "你好"
    assert deltas[1]["text"] == "，"
    assert deltas[2]["text"] == "世界"
    assert len(stop) == 1


# ---------- 2. test_multi_turn_context ----------

@pytest.mark.asyncio
async def test_multi_turn_context(client, mock_anthropic):
    # Round 1
    setup_normal_chat(mock_anthropic, ["回复1"])
    await client.post("/chat", json={
        "session_id": "multi-1",
        "message": "第一句话",
    })

    # Round 2 — need fresh stream mock
    setup_normal_chat(mock_anthropic, ["回复2"])
    await client.post("/chat", json={
        "session_id": "multi-1",
        "message": "第二句话",
    })

    # Verify history
    resp = await client.get("/sessions/multi-1/history")
    assert resp.status_code == 200

    data = resp.json()
    messages = data["messages"]

    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "第一句话"}
    assert messages[1] == {"role": "assistant", "content": "回复1"}
    assert messages[2] == {"role": "user", "content": "第二句话"}
    assert messages[3] == {"role": "assistant", "content": "回复2"}


# ---------- 3. test_tool_use ----------

@pytest.mark.asyncio
async def test_tool_use(client, mock_anthropic):
    setup_tool_use_chat(
        mock_anthropic,
        tool_id="call_weather_1",
        tool_name="get_weather",
        tool_input={"city": "北京"},
        final_chunks=["北京今天22°C，天气晴朗"],
    )

    resp = await client.post("/chat", json={
        "session_id": "tool-1",
        "message": "北京今天天气怎么样？",
    })
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    full_text = "".join(d["text"] for d in deltas)
    assert "22°C" in full_text

    # Verify tool_use and tool_result are in history
    hist_resp = await client.get("/sessions/tool-1/history")
    messages = hist_resp.json()["messages"]

    # user -> assistant(tool_use) -> user(tool_result) -> assistant(final)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in messages[1]["content"])
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "call_weather_1"
    # tool_result content should contain weather data
    tool_result_data = json.loads(messages[2]["content"][0]["content"])
    assert tool_result_data["city"] == "北京"
    assert messages[3]["role"] == "assistant"


# ---------- 4. test_rag_retrieval ----------

@pytest.mark.asyncio
async def test_rag_retrieval(client, mock_anthropic):
    # Upload a test document
    file_content = "本产品支持智能问答、文档检索和自动摘要三大核心功能。"
    upload_resp = await client.post(
        "/knowledge/upload",
        files={"file": ("产品说明.txt", io.BytesIO(file_content.encode("utf-8")), "text/plain")},
    )
    assert upload_resp.status_code == 200
    assert upload_resp.json()["status"] == "success"

    # Capture the system prompt passed to Claude API
    captured_system = {}

    async def capture_create(**kwargs):
        captured_system["value"] = kwargs.get("system", "")
        from tests.conftest import _make_text_block, _make_response
        return _make_response([_make_text_block("基于知识库回答")], "end_turn")

    mock_anthropic.messages.create = capture_create
    from tests.conftest import _make_stream_ctx
    mock_anthropic.messages.stream = lambda **kwargs: _make_stream_ctx(["基于知识库的回答"])

    resp = await client.post("/chat", json={
        "session_id": "rag-1",
        "message": "产品有什么功能？",
        "use_knowledge": True,
    })
    assert resp.status_code == 200

    # Verify knowledge was injected into system prompt
    system_prompt = captured_system["value"]
    assert "智能问答" in system_prompt or "文档检索" in system_prompt or "自动摘要" in system_prompt


# ---------- 5. test_error_handling ----------

@pytest.mark.asyncio
async def test_error_session_not_found(client, mock_anthropic):
    resp = await client.get("/sessions/nonexistent-session/history")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"


@pytest.mark.asyncio
async def test_error_delete_not_found(client, mock_anthropic):
    resp = await client.delete("/sessions/nonexistent-session")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"


@pytest.mark.asyncio
async def test_error_missing_fields(client, mock_anthropic):
    # Missing session_id and message
    resp = await client.post("/chat", json={})
    assert resp.status_code == 422

    # Missing message
    resp = await client.post("/chat", json={"session_id": "x"})
    assert resp.status_code == 422

    # Missing session_id
    resp = await client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 422
