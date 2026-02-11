from app.tools.get_weather import get_weather
from app.tools.calculator import calculator
from app.tools.web_search import web_search

# OpenAI-compatible tool definitions (used by OpenRouter)
TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算器，支持加减乘除和括号运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：2+3*4、(10-3)/2",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索，获取实时网络信息。当用户询问最新资讯、新闻、实时数据等需要联网的问题时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如：最新科技新闻、Python 3.12新特性",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认5条",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# Keep the old name for backward compatibility in /tools route
TOOL_DEFINITIONS = TOOL_DEFINITIONS_OPENAI

_DISPATCH = {
    "get_weather": get_weather,
    "calculator": calculator,
    "web_search": web_search,
}


def dispatch(tool_name: str, tool_input: dict) -> dict:
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**tool_input)
