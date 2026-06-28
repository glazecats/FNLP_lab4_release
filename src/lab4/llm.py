from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


Message = dict[str, str]


@dataclass
class LLMConfig:
    model: str = os.environ.get("LLM_MODEL", "qwen3-8b")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    api_key: str | None = os.environ.get("DASHSCOPE_API_KEY")
    timeout: int = int(os.environ.get("LLM_TIMEOUT", "120"))
    mock: bool = os.environ.get("LAB4_MOCK_LLM", "").lower() in {"1", "true", "yes"}


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> str:
        if self.config.mock:
            return self._mock_response(messages)
        if not self.config.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": bool(enable_thinking),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(4):
            request = urllib.request.Request(self.config.base_url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    raw = response.read().decode("utf-8")
                body = json.loads(raw)
                return body["choices"][0]["message"]["content"]
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                socket.timeout,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep((2**attempt) + random.random())
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    def _mock_response(self, messages: list[Message]) -> str:
        system = messages[0].get("content", "") if messages else ""
        text = "\n".join(message.get("content", "") for message in messages)
        if "你是 RETRIEVAL_QUERY_WRITER" in system:
            return "QUERY: physics chemistry textbook target quantity units"
        if "你是 RAG_CURATOR" in system:
            return "RAG_NOTES:\n- Mock note: use the equation that matches the requested quantity.\n- Check target units."
        if "你是 VERIFIER" in system:
            return "PASS\nFINAL_ANSWER: 1"
        if "你是 ARBITER" in system:
            return "FINAL_ANSWER: 1"
        if "TOOL_CALC_DEMO" in text:
            return "TOOL_CALC: sqrt(4)"
        return "简要推理：mock 模式不调用远端模型。\nFINAL_ANSWER: 1"
