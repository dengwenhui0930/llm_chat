import os
import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Regex for {{if var}}...{{/if}} blocks
_COND_RE = re.compile(
    r"\{\{if\s+(\w+)\}\}\s*\n?(.*?)\{\{/if\}\}",
    re.DOTALL,
)


def _render_conditionals(template: str, variables: dict[str, str]) -> str:
    """Process {{if variable}}...{{/if}} blocks."""
    def replacer(m: re.Match) -> str:
        var_name = m.group(1)
        body = m.group(2)
        if variables.get(var_name):
            return body
        return ""
    return _COND_RE.sub(replacer, template)


def _render_variables(text: str, variables: dict[str, str]) -> str:
    """Replace {{variable}} placeholders with values."""
    for key, value in variables.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def render_template(template: str, variables: dict[str, str]) -> str:
    """Render a prompt template with conditionals and variable substitution."""
    result = _render_conditionals(template, variables)
    result = _render_variables(result, variables)
    # Clean up extra blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def get_current_version() -> str:
    return os.environ.get("PROMPT_VERSION", "v1_default")


def list_versions() -> list[str]:
    versions = []
    for f in sorted(_PROMPTS_DIR.glob("v*.txt")):
        versions.append(f.stem)
    return versions


def load_template(version: str | None = None) -> str:
    version = version or get_current_version()
    path = _PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {version}")
    return path.read_text(encoding="utf-8").strip()
