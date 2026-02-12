import json
import os
import time
from collections.abc import AsyncGenerator

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    APITimeoutError,
    APIError,
)

from app.services.prompt_manager import load_template, render_template
from app.services.retriever import retrieve
from app.tools.tool_registry import TOOL_DEFINITIONS_OPENAI, dispatch

_MODEL = os.environ.get("CHAT_MODEL", "anthropic/claude-sonnet-4")
_MAX_TOKENS = 4096

_SESSION_MAX_COUNT = 1000
_SESSION_TTL_SECONDS = 3600  # 1 hour

# In-memory session storage: session_id -> list of messages
_sessions: dict[str, list[dict]] = {}
_session_last_active: dict[str, float] = {}


# --------------- Session accessor functions ---------------

def get_session(session_id: str) -> list[dict] | None:
    return _sessions.get(session_id)


def remove_session(session_id: str) -> bool:
    removed = _sessions.pop(session_id, None) is not None
    _session_last_active.pop(session_id, None)
    return removed


def get_model() -> str:
    return _MODEL


def set_model(model: str) -> None:
    global _MODEL
    _MODEL = model


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, ts in _session_last_active.items()
        if now - ts > _SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_last_active.pop(sid, None)

    if len(_sessions) > _SESSION_MAX_COUNT:
        sorted_by_time = sorted(_session_last_active.items(), key=lambda x: x[1])
        for sid, _ in sorted_by_time[: len(_sessions) - _SESSION_MAX_COUNT]:
            _sessions.pop(sid, None)
            _session_last_active.pop(sid, None)


class ChatService:
    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _build_system_prompt(self, message: str, use_knowledge: bool) -> str:
        variables: dict[str, str] = {}

        if use_knowledge:
            chunks = retrieve(message)
            if chunks:
                references = "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))
                variables["knowledge_context"] = references

        tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS_OPENAI]
        if tool_names:
            variables["tool_list"] = ", ".join(tool_names)

        template = load_template()
        return render_template(template, variables)

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        use_knowledge: bool = False,
    ) -> AsyncGenerator[str, None]:
        _cleanup_expired_sessions()

        if session_id not in _sessions:
            _sessions[session_id] = []
        _session_last_active[session_id] = time.time()
        history = _sessions[session_id]

        system_prompt = self._build_system_prompt(message, use_knowledge)

        history.append({"role": "user", "content": message})

        # Build messages list with system prompt at the front
        def _build_messages():
            return [{"role": "system", "content": system_prompt}] + history

        try:
            # Tool use loop: non-streaming calls until model stops calling tools
            while True:
                response = await self.client.chat.completions.create(
                    model=_MODEL,
                    max_tokens=_MAX_TOKENS,
                    messages=_build_messages(),
                    tools=TOOL_DEFINITIONS_OPENAI,
                )

                choice = response.choices[0]
                assistant_msg = choice.message

                if not assistant_msg.tool_calls:
                    break

                # Append assistant message with tool_calls to history
                history.append({
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                })

                # Execute each tool and append tool results
                for tc in assistant_msg.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    result = dispatch(tc.function.name, tool_input)
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # Use the non-streaming response directly — no redundant second API call
            full_response = assistant_msg.content or ""
            yield f"data: {json.dumps({'type': 'content_block_delta', 'text': full_response}, ensure_ascii=False)}\n\n"

        except (AuthenticationError, APITimeoutError, APIError):
            # Roll back all messages added in this turn
            while history and history[-1].get("role") != "user":
                history.pop()
            if history:
                history.pop()
            raise

        history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
