import os

from fastapi import APIRouter, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.chat import ChatRequest
from app.services.chat_service import ChatService, get_session, remove_session, get_model, set_model
from app.services.knowledge_service import ALLOWED_EXTENSIONS, delete_file, extract_docx_text, list_files, split_chunks, store_chunks
from app.services.prompt_manager import get_current_version, list_versions, load_template
from app.tools.tool_registry import TOOL_DEFINITIONS_OPENAI

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
    messages = get_session(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": messages}


@router.get("/tools")
async def list_tools():
    tools = []
    for defn in TOOL_DEFINITIONS_OPENAI:
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
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: .txt, .md, .docx")
    raw = await file.read()
    if ext == ".docx":
        content = extract_docx_text(raw)
    else:
        content = raw.decode("utf-8")
    chunks = split_chunks(content)
    store_chunks(file.filename, chunks)
    return {"filename": file.filename, "chunks": len(chunks), "status": "success"}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if not remove_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


# ---------- Knowledge Files ----------

@router.get("/knowledge/files")
async def list_knowledge_files():
    return {"files": list_files()}


@router.delete("/knowledge/files/{filename}")
async def delete_knowledge_file(filename: str):
    if not delete_file(filename):
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "deleted", "filename": filename}


# ---------- Settings ----------

class SettingsUpdate(BaseModel):
    chat_model: str | None = None
    prompt_version: str | None = None


@router.get("/settings")
async def get_settings():
    return {
        "chat_model": get_model(),
        "prompt_version": get_current_version(),
        "available_prompt_versions": list_versions(),
    }


@router.put("/settings")
async def update_settings(body: SettingsUpdate):
    updated = {}
    if body.chat_model is not None:
        set_model(body.chat_model)
        updated["chat_model"] = body.chat_model
    if body.prompt_version is not None:
        versions = list_versions()
        if body.prompt_version not in versions:
            raise HTTPException(status_code=400, detail=f"Unknown prompt version: {body.prompt_version}")
        os.environ["PROMPT_VERSION"] = body.prompt_version
        updated["prompt_version"] = body.prompt_version
    return {"status": "updated", **updated}
