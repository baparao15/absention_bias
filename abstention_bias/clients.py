from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Protocol

import aiohttp

from .config import ModelConfig


class LLMRequestError(RuntimeError):
    def __init__(self, status: int | None, message: str, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def requires_key_change(self) -> bool:
        return self.status in {401, 403, 429}


class ChatClient(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

class OpenAICompatibleChatClient:
    _rate_lock = threading.Lock()
    _last_request_at: dict[str, float] = {}

    def __init__(self, config: ModelConfig):
        if not config.api_key or not config.base_url:
            raise ValueError("ABSTENTION_API_KEY and ABSTENTION_BASE_URL are required for API clients.")
        self.config = config

    async def _wait_for_rate_slot(self) -> None:
        interval = max(0.0, self.config.min_interval_sec)
        if interval <= 0:
            return
        key = f"{self.config.base_url}|{self.config.model_name}|{self.config.api_key[:12]}"
        with self._rate_lock:
            now = time.monotonic()
            last = self._last_request_at.get(key, 0.0)
            wait_sec = max(0.0, interval - (now - last))
            self._last_request_at[key] = now + wait_sec
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)

    def _retry_after_seconds(self, body: str, attempt: int) -> float:
        match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", body, flags=re.IGNORECASE)
        if match:
            return max(float(match.group(1)) + 0.5, self.config.retry_base_sec)
        minute_match = re.search(
            r"try again in (?:(?P<minutes>[0-9]+(?:\.[0-9]+)?)m)?(?:(?P<seconds>[0-9]+(?:\.[0-9]+)?)s)?",
            body,
            flags=re.IGNORECASE,
        )
        if minute_match and (minute_match.group("minutes") or minute_match.group("seconds")):
            minutes = float(minute_match.group("minutes") or 0.0)
            seconds = float(minute_match.group("seconds") or 0.0)
            return max(minutes * 60 + seconds + 1.0, self.config.retry_base_sec)
        return self.config.retry_base_sec * (attempt + 1)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_sec)
        attempts = max(1, self.config.max_retries + 1)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for attempt in range(attempts):
                await self._wait_for_rate_slot()
                async with session.post(self.config.base_url, json=payload) as response:
                    if response.status == 429 and attempt < attempts - 1:
                        body = await response.text()
                        await asyncio.sleep(self._retry_after_seconds(body, attempt))
                        continue
                    if response.status >= 400:
                        body = await response.text()
                        raise LLMRequestError(
                            response.status,
                            f"LLM request failed with HTTP {response.status}.",
                            body[:2000],
                        )
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
        raise LLMRequestError(None, "LLM request failed after retry attempts.")


def build_chat_client(config: ModelConfig) -> ChatClient:
    return OpenAICompatibleChatClient(config)
