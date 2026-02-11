# Xiaohongshu (XHS) Content Agent

AI-powered content creation and posting agent for Xiaohongshu (RedNote).

## Architecture

```
BrowserAgent -> ContentCreator -> Publisher -> END
```

- **BrowserAgent**: Crawls XHS for research using `agent-browser-mcp` (Node.js + Playwright)
- **ContentCreator**: Generates content based on research
- **Publisher**: Posts to XHS via Go MCP server

## Project Structure

```
socialmedia_upload/
├── agent/
│   ├── graph.py            # Main LangGraph workflow
│   ├── main.py             # Entry point
│   ├── prompts/            # Agent prompts
│   └── tools/
│       ├── browser_client.py   # Python client for agent-browser-mcp
│       ├── xhs_mcp.py          # Publisher tools (Go MCP)
│       └── ...
├── browser_mcp/            # Node.js MCP server for browser automation
│   ├── package.json
│   └── src/                # TypeScript source
├── external_xhs_mcp/       # Go MCP server for publishing
├── data/                   # Crawled data output
└── ui/                     # Streamlit UI
```

## Key Files

- `agent/graph.py` - Main workflow definition (`build_graph`)
- `agent/tools/browser_client.py` - Wraps `agent-browser-mcp` for LangChain
- `browser_mcp/` - Node.js based browser automation server
- `external_xhs_mcp/` - Go based publishing server

## Development

### Run Full Agent
```bash
python agent/graph.py
```

### Run UI
```bash
streamlit run ui/app.py
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url  # Optional
```

## Browser Automation

Uses `agent-browser-mcp` (Node.js) which leverages Playwright/Puppeteer for automation:
- **Server**: Located in `browser_mcp/`
- **Client**: `agent/tools/browser_client.py` starts the Node process via `npx`
- **Capabilities**: Natural language interaction with the browser

## Important Notes

- **Browser Agent**: Launches a Node.js sub-process for browser control.
- **Publisher**: Uses a Go MCP server for XHS API interactions.
- **Login**: May require manual login depending on the session state managed by the browser agent.

---

## LLM Chat Service

FastAPI backend for multi-turn Claude conversations, located in `app/`.

### Structure

```
app/
├── main.py              # FastAPI entry point
├── models/chat.py       # Pydantic models (ChatMessage, ChatRequest, ChatResponse)
├── routes/chat.py       # API routes (/chat, /chat/stream, /health)
└── services/chat_service.py  # Anthropic SDK wrapper
prompts/chat_system.md   # Default system prompt
tests/test_chat_service.py  # Unit tests
```

### Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/test_chat_service.py
```

### Environment Variables

- `ANTHROPIC_API_KEY` — Required for Claude API access
