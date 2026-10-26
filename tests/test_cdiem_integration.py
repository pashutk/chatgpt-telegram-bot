"""Network-free regression checks for the cdiem integration.

Run with: python -m unittest discover -s tests -v
"""

import base64
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
import openai
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bot"))

from openai_helper import OpenAIHelper, default_max_tokens
from plugin_manager import PluginManager
from plugins.openai_web_search import OpenAIWebSearchPlugin


def completion(content="Answer", tool_calls=None):
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
        "model": "test", "choices": [{
            "index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Token counting is independent of request serialization. Avoid fetching
        # tiktoken's vocabulary during these otherwise network-free tests.
        encoding = type("Encoding", (), {"encode": lambda self, text: list(text.encode())})()
        self.encoding_patch = patch("openai_helper.tiktoken.encoding_for_model", return_value=encoding)
        self.encoding_patch.start()
        self.addCleanup(self.encoding_patch.stop)

    async def helper(self, handler, model="gpt-5.1", plugins=(), **overrides):
        config = {
            "api_key": "test-key", "model": model, "vision_model": "gpt-6-sol",
            "assistant_prompt": "Answer briefly", "max_tokens": default_max_tokens(model),
            "max_history_size": 100, "max_conversation_age_minutes": 180,
            "temperature": 0.7, "presence_penalty": 0.1, "frequency_penalty": 0.2,
            "reasoning_effort": "low", "n_choices": 1, "bot_language": "en",
            "enable_functions": bool(plugins), "functions_max_consecutive_calls": 3,
            "show_usage": False, "show_plugins_used": False,
            "image_model": "gpt-image-2", "image_quality": "auto", "image_size": "auto",
            "vision_detail": "auto", "vision_max_tokens": 300,
            "vision_prompt": "Describe this", "enable_vision_follow_up_questions": True,
        }
        config.update(overrides)
        helper = OpenAIHelper(config, PluginManager({"plugins": plugins}))
        await helper.client.close()
        helper.client = openai.AsyncOpenAI(
            api_key="test-key", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        self.addAsyncCleanup(helper.client.close)
        return helper

    async def test_reasoning_requests_preserve_fork_models_and_support_unknown_models(self):
        for model in ("gpt-5", "gpt-5.1", "gpt-5.1-codex", "o3", "gpt-6-sol"):
            with self.subTest(model=model):
                requests = []
                def handler(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, json=completion())
                helper = await self.helper(handler, model=model)
                answer, tokens = await helper.get_chat_response(1, "Hello")
                self.assertEqual((answer, tokens), ("Answer", 15))
                body = requests[0]
                self.assertEqual(body["model"], model)
                self.assertEqual(body["max_completion_tokens"], default_max_tokens(model))
                self.assertEqual(body["reasoning_effort"], "low")
                self.assertTrue({"max_tokens", "temperature", "presence_penalty", "frequency_penalty"}.isdisjoint(body))
        self.assertEqual(default_max_tokens("gpt-5.1"), 8192)
        self.assertEqual(default_max_tokens("gpt-6-sol"), 16000)

    async def test_existing_gpt_41_support_retains_legacy_request_parameters(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=completion())
        helper = await self.helper(handler, model="gpt-4.1")
        await helper.get_chat_response(1, "Hello")
        self.assertEqual(requests[0]["max_tokens"], 4096)
        self.assertEqual(requests[0]["temperature"], 0.7)
        self.assertNotIn("reasoning_effort", requests[0])

    async def test_image_generation_decodes_base64_without_legacy_parameters(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"created": 0, "data": [{
                "url": None, "b64_json": base64.b64encode(b"png-data").decode(),
            }]})
        helper = await self.helper(handler)
        self.assertEqual(await helper.generate_image("Draw a cat"), (b"png-data", "auto"))
        self.assertEqual(requests[0]["quality"], "auto")
        self.assertEqual(requests[0]["model"], "gpt-image-2")
        self.assertNotIn("style", requests[0])
        self.assertNotIn("response_format", requests[0])

    async def test_vision_follow_up_selects_vision_model_parameters(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=completion("A blue square"))
        helper = await self.helper(handler, model="gpt-4.1")
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, "PNG")
        await helper.interpret_image(1, image, "Describe it")
        await helper.get_chat_response(1, "What color?")
        self.assertEqual(len(requests), 2)
        for body in requests:
            self.assertEqual(body["model"], "gpt-6-sol")
            self.assertIn("max_completion_tokens", body)
            self.assertNotIn("temperature", body)

    async def test_streaming_with_plugins_does_not_drop_the_first_text_chunk(self):
        def handler(request):
            chunks = []
            for text, finish in (("First ", None), ("second", None), (None, "stop")):
                chunks.append("data: " + json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                    "model": "test", "choices": [{"index": 0, "finish_reason": finish,
                        "delta": {"content": text}}],
                }) + "\n\n")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="".join(chunks) + "data: [DONE]\n\n")
        helper = await self.helper(handler, plugins=["image_generation"])
        answers = [answer async for answer in helper.get_chat_response_stream(1, "Hello")]
        self.assertEqual(answers[-1][0], "First second")

    async def test_image_tool_uses_tool_history_and_reports_progress(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/images/generations"):
                return httpx.Response(200, json={"created": 0, "data": [{
                    "url": "https://example.com/generated.png", "b64_json": None,
                }]})
            return httpx.Response(200, json=completion(None, [{
                "id": "call-image", "type": "function", "function": {
                    "name": "generate_image", "arguments": '{"prompt":"A cat"}',
                },
            }]))
        helper = await self.helper(handler, plugins=["image_generation"])
        progress = []
        async def on_function_call(name):
            progress.append(name)
        answer, _ = await helper.get_chat_response(1, "Draw a cat", on_function_call)
        self.assertEqual(json.loads(answer)["direct_result"]["value"], "https://example.com/generated.png")
        self.assertEqual(progress, ["generate_image"])
        self.assertEqual(helper.plugin_manager.get_progress_message_key("generate_image"), "image_generation_progress")
        self.assertIn("tools", requests[0][1])
        self.assertNotIn("functions", requests[0][1])
        self.assertEqual(helper.conversations[1][-1]["role"], "tool")
        self.assertEqual(helper.conversations[1][-1]["tool_call_id"], "call-image")
        self.assertEqual(helper.get_conversation_stats(1)[0], 4)

    async def test_plugin_errors_are_returned_to_the_model(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return httpx.Response(200, json=completion(None, [{
                    "id": "call-bad", "type": "function", "function": {
                        "name": "generate_image", "arguments": "not-json",
                    },
                }]))
            return httpx.Response(200, json=completion("The image could not be generated"))
        helper = await self.helper(handler, plugins=["image_generation"])
        with self.assertLogs(level="ERROR"):
            answer, _ = await helper.get_chat_response(1, "Draw")
        self.assertEqual(answer, "The image could not be generated")
        result = requests[1]["messages"][-1]
        self.assertEqual(result["role"], "tool")
        self.assertIn("error", json.loads(result["content"]))

    async def test_openai_web_search_uses_responses_endpoint(self):
        requests = []
        def handler(request):
            requests.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"output": [{"type": "message", "content": [
                {"type": "output_text", "text": "Answer with sources"},
            ]}]})
        helper = await self.helper(handler)
        result = await OpenAIWebSearchPlugin().execute("web_search", helper, query="Recent news")
        self.assertEqual(result, {"answer": "Answer with sources"})
        self.assertEqual(requests[0][0], "/v1/responses")
        self.assertEqual(requests[0][1]["tools"], [{"type": "web_search"}])


if __name__ == "__main__":
    unittest.main()
