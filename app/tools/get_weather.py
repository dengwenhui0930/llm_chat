_MOCK_WEATHER: dict[str, dict] = {
    "北京": {"city": "北京", "temperature": "22°C", "condition": "晴"},
    "上海": {"city": "上海", "temperature": "26°C", "condition": "多云"},
    "广州": {"city": "广州", "temperature": "30°C", "condition": "阵雨"},
    "深圳": {"city": "深圳", "temperature": "29°C", "condition": "多云转晴"},
    "杭州": {"city": "杭州", "temperature": "24°C", "condition": "阴"},
    "成都": {"city": "成都", "temperature": "20°C", "condition": "小雨"},
}

_DEFAULT_WEATHER = {"temperature": "25°C", "condition": "晴"}


def get_weather(city: str) -> dict:
    if city in _MOCK_WEATHER:
        return _MOCK_WEATHER[city]
    return {"city": city, **_DEFAULT_WEATHER}
