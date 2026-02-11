from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import AuthenticationError, APITimeoutError, APIError

from app.routes.chat import router as chat_router

app = FastAPI(title="LLM Chat Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=500,
        content={"detail": "API Key is not configured or invalid"},
    )


@app.exception_handler(APITimeoutError)
async def timeout_handler(request: Request, exc: APITimeoutError):
    return JSONResponse(
        status_code=502,
        content={"detail": f"LLM API error: {exc}"},
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=502,
        content={"detail": f"LLM API error: {exc}"},
    )


app.include_router(chat_router)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
async def serve_frontend():
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
