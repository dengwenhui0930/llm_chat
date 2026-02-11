# LLM Chat Service

基于 Claude API 的多轮对话后端服务，支持 **Tool Use**（工具调用）和 **RAG**（检索增强生成）。

## 技术栈

- **Python 3.11+**
- **FastAPI** — 异步 Web 框架
- **Anthropic SDK** — Claude API 官方 Python 客户端
- **jieba** — 中文分词，用于 RAG 检索
- **uvicorn** — ASGI 服务器

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

或在项目根目录创建 `.env` 文件：

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
PROMPT_VERSION=v1_default        # 可选，默认 v1_default
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. 运行测试

```bash
pytest -v
```

## 架构

```
                         POST /chat
                            |
                            v
                     +-------------+
                     |   FastAPI   |
                     |   Router    |
                     +------+------+
                            |
               +------------+------------+
               |                         |
               v                         v
      use_knowledge?              Prompt Manager
        true / false              (模板渲染)
               |                         |
     +---------+---------+               |
     |                   |               v
     v                   v          System Prompt
  Retriever          (skip)              |
  (jieba+TF-IDF)                        |
     |                                   |
     v                                   |
  知识注入 system prompt  <--------------+
                            |
                            v
                  +-------------------+
                  |   Claude API      |
                  |   (streaming)     |
                  +--------+----------+
                           |
              stop_reason == "tool_use" ?
                    /            \
                  yes             no
                  /                \
           +-----------+      +----------+
           | Tool Use  |      | 直接返回  |
           | dispatch  |      | SSE 流   |
           +-----------+      +----------+
                |
                v
           执行工具函数
           (天气/计算器)
                |
                v
          tool_result 回传
          Claude API 继续
                |
                v
           最终 SSE 流返回
```

## API 接口文档

### POST /chat

多轮对话接口，返回 SSE 流。

**请求：**
```json
{
  "session_id": "user-123",
  "message": "北京今天天气怎么样？",
  "use_knowledge": false
}
```

**响应（SSE）：**
```
data: {"type": "content_block_delta", "text": "北京"}
data: {"type": "content_block_delta", "text": "今天"}
data: {"type": "content_block_delta", "text": "晴朗"}
data: {"type": "message_stop"}
```

---

### GET /sessions/{session_id}/history

获取会话历史消息。

**响应：**
```json
{
  "session_id": "user-123",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
  ]
}
```

**错误：** `404` — `{"detail": "Session not found"}`

---

### DELETE /sessions/{session_id}

删除会话。

**响应：** `204 No Content`

**错误：** `404` — `{"detail": "Session not found"}`

---

### GET /tools

返回已注册的工具列表。

**响应：**
```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "查询指定城市的天气信息",
      "parameters": {"city": {"type": "string", "required": true}}
    },
    {
      "name": "calculator",
      "description": "数学计算器，支持加减乘除和括号运算",
      "parameters": {"expression": {"type": "string", "required": true}}
    }
  ]
}
```

---

### POST /knowledge/upload

上传知识文件（.txt / .md），用于 RAG 检索。

**请求：** `multipart/form-data`，字段名 `file`

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@产品手册.txt"
```

**响应：**
```json
{
  "filename": "产品手册.txt",
  "chunks": 12,
  "status": "success"
}
```

**错误：** `400` — 文件格式不是 .txt 或 .md

---

### GET /prompts

返回当前 prompt 模板信息。

**响应：**
```json
{
  "current_version": "v1_default",
  "available_versions": ["v1_default", "v2_professional"],
  "current_template": "你是一个智能助手，名叫 SmartBot。\n\n..."
}
```

---

### GET /health

健康检查。

**响应：**
```json
{"status": "ok", "service": "llm-chat"}
```

## 项目目录结构

```
llm-chat/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 入口，CORS，异常处理
│   ├── models/
│   │   └── chat.py              # Pydantic 模型 (ChatMessage, ChatRequest)
│   ├── routes/
│   │   └── chat.py              # API 路由定义
│   ├── services/
│   │   ├── chat_service.py      # Claude API 调用，Tool Use 循环，会话管理
│   │   ├── knowledge_service.py # 知识文件分块与存储
│   │   ├── prompt_manager.py    # Prompt 模板加载、条件渲染、版本管理
│   │   └── retriever.py         # jieba 分词 + TF-IDF 检索
│   └── tools/
│       ├── calculator.py        # 安全数学计算器（ast 解析）
│       ├── get_weather.py       # 天气查询（mock 数据）
│       └── tool_registry.py     # 工具注册、JSON Schema 定义、dispatch
├── prompts/
│   ├── chat_system.md           # 旧版默认 system prompt
│   ├── rag_system.md            # 旧版 RAG prompt
│   ├── v1_default.txt           # 模板 v1：通用助手
│   └── v2_professional.txt      # 模板 v2：企业级专业助手
├── tests/
│   ├── conftest.py              # 共用 fixtures，mock Anthropic client
│   ├── test_api.py              # 端到端 API 测试（httpx + FastAPI）
│   ├── test_chat_service.py     # ChatService 单元测试
│   ├── test_knowledge.py        # 知识分块测试
│   ├── test_prompt_manager.py   # 模板渲染测试
│   ├── test_retriever.py        # 检索测试
│   └── test_tools.py            # 工具函数测试
├── .env                         # 环境变量（不提交到 git）
├── requirements.txt             # Python 依赖
└── README.md
```

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | 是 | Claude API 密钥 |
| `PROMPT_VERSION` | 否 | Prompt 模板版本，默认 `v1_default` |
