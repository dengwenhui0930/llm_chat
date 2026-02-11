from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import anthropic

from app.routes.chat import router as chat_router

app = FastAPI(title="LLM Chat Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(anthropic.AuthenticationError)
async def anthropic_auth_error_handler(request: Request, exc: anthropic.AuthenticationError):
    return JSONResponse(
        status_code=500,
        content={"detail": "Anthropic API Key is not configured or invalid"},
    )


@app.exception_handler(anthropic.APITimeoutError)
async def anthropic_timeout_handler(request: Request, exc: anthropic.APITimeoutError):
    return JSONResponse(
        status_code=502,
        content={"detail": f"Claude API error: {exc}"},
    )


@app.exception_handler(anthropic.APIError)
async def anthropic_api_error_handler(request: Request, exc: anthropic.APIError):
    return JSONResponse(
        status_code=502,
        content={"detail": f"Claude API error: {exc}"},
    )


app.include_router(chat_router)
