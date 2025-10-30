from typing import List, Dict
from config.config import (
    PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL
)

class LLMClient:
    """
    Unified chat interface across providers:
    - Google Gemini (default; supports 2.0 models like 'models/gemini-2.0-flash-lite')
    - OpenAI
    - Groq
    - Hugging Face Inference API
    """
    def __init__(self, provider: str = None, model: str = None):
        self.provider = (provider or PROVIDER).lower()
        self.model = model
        self.client = None
        self.hf_headers = None

        if self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing.")
            try:
                from groq import Groq  # type: ignore
            except Exception as e:
                raise ImportError("groq package not installed. pip install groq") from e
            self.client = Groq(api_key=GROQ_API_KEY)

        elif self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing.")
            try:
                import google.generativeai as genai  # type: ignore
            except Exception as e:
                raise ImportError("google-generativeai not installed. pip install google-generativeai") from e
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai  # we keep the module for later use

        elif self.provider == "huggingface":
            if not HUGGINGFACE_API_KEY:
                raise ValueError("HUGGINGFACE_API_KEY is missing.")
            try:
                import requests  # noqa: F401
            except Exception as e:
                raise ImportError("requests is required for Hugging Face API. pip install requests") from e
            self.hf_headers = { "Authorization": f"Bearer {HUGGINGFACE_API_KEY}" }

        else:  # openai
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing.")
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise ImportError("openai package is not installed. pip install openai>=1.30.0") from e
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(self, messages: List[Dict], temperature: float = 0.2) -> str:
        if self.provider == "groq":
            try:
                resp = self.client.chat.completions.create(
                    model=self.model or GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"[LLM Error - Groq] {e}"

        elif self.provider == "gemini":
            try:
                # Build a simple text prompt from messages (Gemini supports multi-turn but this is sufficient)
                def rp(m): return {"system":"[SYSTEM]","user":"[USER]","assistant":"[ASSISTANT]"}.get(m.get("role","user"),"[USER]")
                prompt = "\n".join(f"{rp(m)} {m.get('content','')}" for m in messages)
                model_id = self.model or GEMINI_MODEL
                model = self.client.GenerativeModel(model_id)
                resp = model.generate_content(prompt, generation_config={"temperature": temperature})
                return getattr(resp, "text", "") or str(resp)
            except Exception as e:
                return f"[LLM Error - Gemini] {e}"

        elif self.provider == "huggingface":
            try:
                import requests, json
                model = self.model or HUGGINGFACE_MODEL
                url = f"https://api-inference.huggingface.co/models/{model}"
                user_content = ""
                for m in messages[::-1]:
                    if m.get("role") == "user":
                        user_content = m.get("content", "")
                        break
                payload = {"inputs": user_content, "parameters": {"temperature": temperature}}
                r = requests.post(url, headers=self.hf_headers, json=payload, timeout=120)
                data = r.json()
                if isinstance(data, list) and len(data) and isinstance(data[0], dict):
                    if "generated_text" in data[0]:
                        return data[0]["generated_text"]
                if isinstance(data, dict):
                    if "generated_text" in data: return data["generated_text"]
                    if "error" in data: return f"[HF Error] {data['error']}"
                return str(data)
            except Exception as e:
                return f"[LLM Error - HF] {e}"

        else:  # openai
            try:
                resp = self.client.chat.completions.create(
                    model=self.model or OPENAI_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"[LLM Error - OpenAI] {e}"
