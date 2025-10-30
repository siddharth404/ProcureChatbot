from typing import List, Dict
from config.config import OPENAI_API_KEY, OPENAI_MODEL
import os

# OpenAI SDK v1+
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class LLMClient:
    def __init__(self, model: str = None):
        self.model = model or OPENAI_MODEL
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing.")
        if OpenAI is None:
            raise ImportError("openai package is not installed.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(self, messages: List[Dict], temperature: float = 0.2) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM Error] {e}"
