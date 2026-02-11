import os

from fastapi import APIRouter, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest
from app.services.chat_service import ChatService, _sessions
from app.services.knowledge_service import ALLOWED_EXTENSIONS, split_chunks, store_chunks
from app.services.prompt_manager import get_current_version, list_versions, load_template
from app.tools.tool_registry import TOOL_DEFINITIONS

router = APIRouter()

try:
    _service = ChatService()
except ValueError:
    _service = None


@router.get("/health")
async def health():
    return {"status": "ok", "service": "llm-chat"}


@router.post("/chat")
async def chat(request: ChatRequest):
    if _service is None:
        raise HTTPException(status_code=500, detail="API Key is not configured or invalid")

    return StreamingResponse(
        _service.chat_stream(
            session_id=request.session_id,
            message=request.message,
            use_knowledge=request.use_knowledge,
        ),
        media_type="text/event-stream",
    )


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": _sessions[session_id]}


@router.get("/tools")
async def list_tools():
    tools = []
    for defn in TOOL_DEFINITIONS:
        func = defn["function"]
        schema = func["parameters"]
        required = set(schema.get("required", []))
        parameters = {
            name: {"type": prop["type"], "required": name in required}
            for name, prop in schema["properties"].items()
        }
        tools.append({
            "name": func["name"],
            "description": func["description"],
            "parameters": parameters,
        })
    return {"tools": tools}


@router.get("/prompts")
async def list_prompts():
    current = get_current_version()
    return {
        "current_version": current,
        "available_versions": list_versions(),
        "current_template": load_template(current),
    }


@router.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Only .txt and .md are allowed")
    content = (await file.read()).decode("utf-8")
    chunks = split_chunks(content)
    store_chunks(file.filename, chunks)
    return {"filename": file.filename, "chunks": len(chunks), "status": "success"}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
