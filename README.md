# Xiaohongshu Content Agent with MCP

This project implements an AI agent that generates content and posts it to Xiaohongshu (RedNote) using the [xpzouying/xiaohongshu-mcp](https://github.com/xpzouying/xiaohongshu-mcp) server.

## Components

1.  **MCP Server (`external_xhs_mcp/`)**: A Go-based MCP server that controls a browser to automate posting.
2.  **Agent Client (`agent/main.py`)**: A Python client that orchestrates content generation and communicates with the MCP server via SSE.

## Prerequisites

*   **Python 3.10+**
*   **Go 1.21+**: Required to run the MCP server.
*   **Chrome/Chromium Browser**: Required for automation.

## Setup

1.  **Install Python Dependencies**:
    ```bash
    pip install mcp requests
    ```

2.  **Initialize External Submodule (if needed)**:
    If `external_xhs_mcp` is empty, ensure it's cloned.

## Usage

### 1. Run the Agent
The agent script attempts to automatically start the Go MCP server if it's not running.

```bash
python agent/main.py
```

### 2. Manual Login (First Time Only)
*   When the agent starts, it will launch a browser window (controlled by the Go server).
*   **You must manually log in to Xiaohongshu** in this window.
*   The session/cookies will be saved in `external_xhs_mcp/cookies` or `data/` for future runs.

### 3. Posting Content
Follow the interactive prompts:
1.  Enter a topic to generate a title/content (mocked).
2.  Provide an **absolute path** to an image.
3.  Confirm to post.

## System Architecture

```mermaid
graph LR
    User --> Agent[Agent Client (Python)]
    Agent -- "HTTP/SSE" --> MCP[MCP Server (Go)]
    MCP -- "Rod/Chrome" --> Browser[Chromium Browser]
    Browser --> XHS[Xiaohongshu]
```

## Troubleshooting

*   **Server fails to start**: Check if you have Go installed (`go version`).
*   **Connection Refused**: Ensure port `18060` is not in use.
*   **Login Issues**: If the browser doesn't open or you can't log in, try running the server manually in `external_xhs_mcp`: `go run . -headless=false`.

---

## LLM Chat Service

A FastAPI-based multi-turn chat backend powered by Claude API, located in `app/`.

### Setup

```bash
pip install -r requirements.txt
```

Add `ANTHROPIC_API_KEY` to your `.env` file:
```
ANTHROPIC_API_KEY=your_key
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

### API Endpoints

| Method | Path           | Description              |
|--------|----------------|--------------------------|
| GET    | `/health`      | Health check             |
| POST   | `/chat`        | Multi-turn chat          |
| POST   | `/chat/stream` | Streaming chat (SSE)     |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Tests

```bash
pytest tests/test_chat_service.py
```
