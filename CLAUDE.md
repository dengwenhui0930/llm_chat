# LLM Chat Service

基于 OpenAI 兼容 API（OpenRouter）的多轮对话后端，支持 Tool Use 和 RAG。

## 命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --port 8000

# 运行全部测试
pytest -v

# 运行单个测试文件
pytest tests/test_api.py -v
pytest tests/test_chat_service.py -v
pytest tests/test_tools.py -v
pytest tests/test_knowledge.py -v
pytest tests/test_retriever.py -v
pytest tests/test_prompt_manager.py -v
```

## 项目结构

```
app/
├── main.py                  # FastAPI 入口
├── models/chat.py           # Pydantic 模型
├── routes/chat.py           # API 路由
├── services/
│   ├── chat_service.py      # OpenAI API 调用 + Tool Use 循环 + 会话管理
│   ├── knowledge_service.py # 知识分块存储 + IDF 缓存标记
│   ├── prompt_manager.py    # Prompt 模板渲染
│   └── retriever.py         # jieba + TF-IDF 检索（带缓存）
└── tools/
    ├── calculator.py        # 计算器工具
    ├── get_weather.py       # 天气工具
    ├── web_search.py        # 联网搜索工具（百度 AI Search）
    └── tool_registry.py     # 工具注册 + dispatch
prompts/                     # Prompt 模板文件
tests/                       # 测试（53 个用例）
```

## 环境变量

- `OPENROUTER_API_KEY` — 必填，OpenRouter API 密钥
- `OPENROUTER_BASE_URL` — 可选，API 地址（默认 `https://openrouter.ai/api/v1`）
- `CHAT_MODEL` — 可选，模型名称（默认 `anthropic/claude-sonnet-4`）
- `PROMPT_VERSION` — 可选，模板版本（默认 `v1_default`）
- `BAIDU_SEARCH_API_KEY` — 可选，百度 AI Search API 密钥

## API 接口

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | 多轮对话（SSE 流式） |
| GET | `/sessions/{id}/history` | 查看会话历史 |
| DELETE | `/sessions/{id}` | 删除会话 |
| GET | `/tools` | 工具列表 |
| POST | `/knowledge/upload` | 上传知识文件 |
| GET | `/knowledge/files` | 查看已上传文件 |
| DELETE | `/knowledge/files/{filename}` | 删除知识文件 |
| GET | `/prompts` | Prompt 模板信息 |
| GET | `/settings` | 查看设置 |
| PUT | `/settings` | 修改模型/模板版本 |
| GET | `/health` | 健康检查 |

## 关键设计

- **会话管理**：内存 dict，key 为 session_id，支持 TTL 自动过期（1 小时）和最大数量限制（1000）
- **Tool Use**：非流式循环调用直到 stop_reason != tool_use，直接使用最终响应
- **RAG**：jieba 分词 + TF-IDF 计算相关性，Top 3 chunk 注入 system prompt，IDF 结果缓存
- **Prompt 模板**：支持 `{{if var}}...{{/if}}` 条件语法和 `{{var}}` 变量替换
