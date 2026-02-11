import json
import os
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

# In-memory session storage: session_id -> list of messages
_sessions: dict[str, list[dict]] = {}


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
        if session_id not in _sessions:
            _sessions[session_id] = []
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

            # Final streaming call
            full_response = ""
            stream = await self.client.chat.completions.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=_build_messages(),
                tools=TOOL_DEFINITIONS_OPENAI,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_response += delta.content
                    yield f"data: {json.dumps({'type': 'content_block_delta', 'text': delta.content}, ensure_ascii=False)}\n\n"

        except (AuthenticationError, APITimeoutError, APIError):
            # Roll back all messages added in this turn
            while history and history[-1].get("role") != "user":
                history.pop()
            if history:
                history.pop()
            raise

        history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
