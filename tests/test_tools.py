from app.tools.get_weather import get_weather
from app.tools.calculator import calculator
from app.tools.tool_registry import TOOL_DEFINITIONS_OPENAI, dispatch


class TestGetWeather:
    def test_known_city(self):
        result = get_weather("北京")
        assert result == {"city": "北京", "temperature": "22°C", "condition": "晴"}

    def test_another_known_city(self):
        result = get_weather("深圳")
        assert result["city"] == "深圳"
        assert "temperature" in result

    def test_unknown_city_returns_default(self):
        result = get_weather("拉萨")
        assert result["city"] == "拉萨"
        assert result["temperature"] == "25°C"
        assert result["condition"] == "晴"


class TestCalculator:
    def test_addition(self):
        assert calculator("2+3")["result"] == 5

    def test_complex_expression(self):
        assert calculator("2+3*4")["result"] == 14

    def test_parentheses(self):
        assert calculator("(10-3)/2")["result"] == 3.5

    def test_negative(self):
        assert calculator("-5+3")["result"] == -2

    def test_division_by_zero(self):
        result = calculator("1/0")
        assert "error" in result

    def test_invalid_expression(self):
        result = calculator("abc")
        assert "error" in result


class TestToolRegistry:
    def test_definitions_format(self):
        for defn in TOOL_DEFINITIONS_OPENAI:
            assert defn["type"] == "function"
            func = defn["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            schema = func["parameters"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_dispatch_weather(self):
        result = dispatch("get_weather", {"city": "上海"})
        assert result["city"] == "上海"

    def test_dispatch_calculator(self):
        result = dispatch("calculator", {"expression": "1+1"})
        assert result["result"] == 2

    def test_dispatch_unknown_tool(self):
        result = dispatch("nonexistent", {})
        assert "error" in result
