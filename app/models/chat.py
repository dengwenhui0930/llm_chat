from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for conversation context")
    message: str = Field(..., description="User message content")
    use_knowledge: bool = Field(default=False, description="Whether to use RAG knowledge retrieval")
