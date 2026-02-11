import os
from unittest.mock import patch

from app.services.prompt_manager import (
    render_template,
    load_template,
    list_versions,
    get_current_version,
)


class TestRenderTemplate:
    def test_variable_replacement(self):
        tpl = "Hello {{name}}, welcome!"
        result = render_template(tpl, {"name": "Alice"})
        assert result == "Hello Alice, welcome!"

    def test_conditional_true(self):
        tpl = "Start.\n{{if ctx}}Content: {{ctx}}{{/if}}\nEnd."
        result = render_template(tpl, {"ctx": "data"})
        assert "Content: data" in result
        assert "{{if" not in result

    def test_conditional_false(self):
        tpl = "Start.\n{{if ctx}}Content: {{ctx}}{{/if}}\nEnd."
        result = render_template(tpl, {})
        assert "Content" not in result
        assert "Start." in result
        assert "End." in result

    def test_conditional_empty_string(self):
        tpl = "A\n{{if x}}show{{/if}}\nB"
        result = render_template(tpl, {"x": ""})
        assert "show" not in result

    def test_multiple_conditionals(self):
        tpl = "{{if a}}A{{/if}} {{if b}}B{{/if}}"
        result = render_template(tpl, {"a": "yes", "b": ""})
        assert "A" in result
        assert "B" not in result

    def test_v1_template_with_knowledge(self):
        tpl = load_template("v1_default")
        result = render_template(tpl, {"knowledge_context": "[1] some info"})
        assert "SmartBot" in result
        assert "[1] some info" in result
        assert "参考资料" in result

    def test_v1_template_without_knowledge(self):
        tpl = load_template("v1_default")
        result = render_template(tpl, {})
        assert "SmartBot" in result
        assert "参考资料" not in result

    def test_v2_template_with_tools(self):
        tpl = load_template("v2_professional")
        result = render_template(tpl, {"tool_list": "get_weather, calculator"})
        assert "get_weather, calculator" in result
        assert "企业级" in result


class TestPromptManager:
    def test_list_versions(self):
        versions = list_versions()
        assert "v1_default" in versions
        assert "v2_professional" in versions

    def test_default_version(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMPT_VERSION", None)
            assert get_current_version() == "v1_default"

    def test_custom_version(self):
        with patch.dict(os.environ, {"PROMPT_VERSION": "v2_professional"}):
            assert get_current_version() == "v2_professional"

    def test_load_nonexistent_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_template("v99_nonexistent")
