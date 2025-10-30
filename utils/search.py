from typing import List, Dict
from config.config import SERPAPI_KEY, SEARCH_PROVIDER

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    results: List[Dict] = []
    try:
        if SEARCH_PROVIDER == "serpapi" and SERPAPI_KEY:
            from serpapi import GoogleSearch
            params = {
                "engine": "google",
                "q": query,
                "num": max_results,
                "api_key": SERPAPI_KEY,
                "hl": "en"
            }
            search = GoogleSearch(params)
            data = search.get_dict()
            for item in (data.get("organic_results") or [])[:max_results]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })
        else:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title"),
                        "link": r.get("href"),
                        "snippet": r.get("body"),
                    })
    except Exception as e:
        results.append({"title": "Search Error", "link": "", "snippet": str(e)})
    return results
