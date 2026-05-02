from __future__ import annotations

import requests
from requests import HTTPError
from openai import AuthenticationError, OpenAI, RateLimitError

from rag_bot.config import Settings
from rag_bot.prompts import SYSTEM_PROMPT


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        if self.settings.llm_provider == "ollama":
            return self._generate_ollama(prompt)
        if self.settings.llm_provider == "openai":
            return self._generate_openai(prompt)
        raise ValueError("LLM_PROVIDER must be either 'ollama' or 'openai'.")

    def _generate_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start Ollama and run "
                f"'ollama pull {self.settings.ollama_model}', or set LLM_PROVIDER=openai in .env."
            ) from exc
        try:
            response.raise_for_status()
        except HTTPError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                "Ollama returned an error. Make sure the model is downloaded with "
                f"'ollama pull {self.settings.ollama_model}'."
                + (f" Details: {detail}" if detail else "")
            ) from exc
        return response.json()["message"]["content"].strip()

    def _generate_openai(self, prompt: str) -> str:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        client = OpenAI(api_key=self.settings.openai_api_key)
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in .env and use a valid key."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI says this account/project has no available quota. Add billing credits or use Ollama locally."
            ) from exc
        content = response.choices[0].message.content
        return (content or "").strip()
