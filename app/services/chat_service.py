import json
import os
from collections.abc import AsyncGenerator

import anthropic

from app.services.prompt_manager import load_template, render_template
from app.services.retriever import retrieve
from app.tools.tool_registry import TOOL_DEFINITIONS, dispatch

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 4096

# In-memory session storage: session_id -> list of messages
_sessions: dict[str, list[dict]] = {}


class ChatService:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _build_system_prompt(self, message: str, use_knowledge: bool) -> str:
        variables: dict[str, str] = {}

        if use_knowledge:
            chunks = retrieve(message)
            if chunks:
                references = "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))
                variables["knowledge_context"] = references

        tool_names = [t["name"] for t in TOOL_DEFINITIONS]
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

        system = self._build_system_prompt(message, use_knowledge)

        history.append({"role": "user", "content": message})

        try:
            # Tool use loop: non-streaming calls until Claude stops calling tools
            while True:
                response = await self.client.messages.create(
                    model=_MODEL,
                    max_tokens=_MAX_TOKENS,
                    system=system,
                    messages=history,
                    tools=TOOL_DEFINITIONS,
                )

                if response.stop_reason != "tool_use":
                    break

                # Append the full assistant message (may contain text + tool_use blocks)
                history.append({"role": "assistant", "content": _serialize_content(response.content)})

                # Execute each tool and build tool_result blocks
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = dispatch(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                history.append({"role": "user", "content": tool_results})

            # Final streaming call — Claude is ready to give a text answer
            full_response = ""
            async with self.client.messages.stream(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=history,
                tools=TOOL_DEFINITIONS,
            ) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'type': 'content_block_delta', 'text': text}, ensure_ascii=False)}\n\n"

        except (anthropic.AuthenticationError, anthropic.APITimeoutError, anthropic.APIError):
            while history and history[-1].get("role") != "user":
                history.pop()
            if history:
                history.pop()
            raise

        history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"


def _serialize_content(content_blocks) -> list[dict]:
    """Convert SDK content blocks to plain dicts for session history."""
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return result
