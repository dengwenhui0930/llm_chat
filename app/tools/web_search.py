"""Web search tool using Baidu AI Search V2 (OpenAI-compatible API)."""

import os

from openai import OpenAI


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web via Baidu AI Search and return results."""
    api_key = os.getenv("BAIDU_SEARCH_API_KEY", "")
    if not api_key:
        return {"query": query, "error": "BAIDU_SEARCH_API_KEY not configured"}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://qianfan.baidubce.com/v2/ai_search",
        )
        resp = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": query}],
        )
        answer = resp.choices[0].message.content if resp.choices else ""

        # Extract search results from response if available
        results = []
        if hasattr(resp, "search_results") and resp.search_results:
            for i, r in enumerate(resp.search_results[:max_results]):
                results.append({
                    "title": getattr(r, "title", ""),
                    "url": getattr(r, "url", ""),
                    "snippet": getattr(r, "snippet", ""),
                })

        return {
            "query": query,
            "answer": answer,
            "results": results,
        }
    except Exception as e:
        return {"query": query, "error": str(e)}
