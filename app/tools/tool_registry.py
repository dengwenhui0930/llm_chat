from app.tools.get_weather import get_weather
from app.tools.calculator import calculator

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
]

# Keep the old name for backward compatibility in /tools route
TOOL_DEFINITIONS = TOOL_DEFINITIONS_OPENAI

_DISPATCH = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def dispatch(tool_name: str, tool_input: dict) -> dict:
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**tool_input)
