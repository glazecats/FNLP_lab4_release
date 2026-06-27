from __future__ import annotations

import json
import os
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "qwen3-8b")
    base_url: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key: str | None = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "0") == "1"
    mock: bool = os.getenv("LAB4_MOCK_LLM", "0") == "1"


class ChatClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        if self.config.mock:
            return "推理轨迹占位。\nFINAL_ANSWER: 0"
        if not self.config.api_key:
            raise RuntimeError(
                "Missing API key. Set DASHSCOPE_API_KEY for Bailian/DashScope Qwen3-8B, "
                "or set LAB4_MOCK_LLM=1 for a local smoke test."
            )

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "enable_thinking": self.config.enable_thinking,
        }
        if self.config.enable_thinking:
            payload["stream"] = True
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    if self.config.enable_thinking:
                        return self._read_stream(response)
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {error_body}")
                time.sleep(2**attempt)
            except (urllib.error.URLError, http.client.HTTPException, OSError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            "LLM call failed after retries.\n"
            f"Endpoint: {endpoint}\n"
            f"Model: {self.config.model}\n"
            f"Last error: {last_error}\n"
            "请检查 DASHSCOPE_API_KEY、LLM_BASE_URL、网络连接，以及是否设置了失效的 HTTP/HTTPS 代理。"
        ) from last_error

    def _read_stream(self, response) -> str:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)
        reasoning_text = "".join(reasoning_parts).strip()
        content_text = "".join(content_parts).strip()
        if reasoning_text and content_text:
            return reasoning_text + "\n" + content_text
        return content_text or reasoning_text
