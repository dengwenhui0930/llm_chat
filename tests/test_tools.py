import os
from unittest.mock import patch, MagicMock

from app.tools.get_weather import get_weather
from app.tools.calculator import calculator
from app.tools.web_search import web_search
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


class TestWebSearch:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BAIDU_SEARCH_API_KEY", None)
            result = web_search("test query")
            assert result["query"] == "test query"
            assert "error" in result
            assert "not configured" in result["error"]

    def test_successful_search(self):
        mock_message = MagicMock()
        mock_message.content = "搜索结果摘要"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.search_results = None

        with patch.dict(os.environ, {"BAIDU_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.web_search.OpenAI") as mock_cls:
                mock_cls.return_value.chat.completions.create.return_value = mock_resp
                result = web_search("Python 最新版本")

        assert result["query"] == "Python 最新版本"
        assert result["answer"] == "搜索结果摘要"
        assert result["results"] == []

    def test_api_error_handling(self):
        from openai import APIError

        with patch.dict(os.environ, {"BAIDU_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.web_search.OpenAI") as mock_cls:
                mock_cls.return_value.chat.completions.create.side_effect = APIError(
                    message="Service unavailable",
                    request=MagicMock(),
                    body=None,
                )
                result = web_search("test")

        assert result["query"] == "test"
        assert "error" in result

    def test_max_results_parameter(self):
        mock_message = MagicMock()
        mock_message.content = "结果"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        # Simulate search_results attribute
        mock_results = []
        for i in range(5):
            r = MagicMock()
            r.title = f"Title {i}"
            r.url = f"https://example.com/{i}"
            r.snippet = f"Snippet {i}"
            mock_results.append(r)
        mock_resp.search_results = mock_results

        with patch.dict(os.environ, {"BAIDU_SEARCH_API_KEY": "test-key"}):
            with patch("app.tools.web_search.OpenAI") as mock_cls:
                mock_cls.return_value.chat.completions.create.return_value = mock_resp
                result = web_search("test", max_results=3)

        assert len(result["results"]) == 3
        assert result["results"][0]["title"] == "Title 0"


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
