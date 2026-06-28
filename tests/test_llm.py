import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.llm import LLMClient, LLMConfig, _optional_int_from_env


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "FINAL_ANSWER: 1"}}]}).encode("utf-8")


class LLMTests(unittest.TestCase):
    def test_chat_payload_includes_seed(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = LLMClient(LLMConfig(api_key="test-key", seed=123, mock=False))
        with patch("urllib.request.urlopen", fake_urlopen):
            response = client.chat([{"role": "user", "content": "x"}], max_tokens=8)

        self.assertEqual(response, "FINAL_ANSWER: 1")
        self.assertEqual(captured["payload"]["seed"], 123)

    def test_optional_seed_env_can_disable_seed(self) -> None:
        with patch.dict("os.environ", {"LLM_SEED": ""}):
            self.assertIsNone(_optional_int_from_env("LLM_SEED", 0))


if __name__ == "__main__":
    unittest.main()
