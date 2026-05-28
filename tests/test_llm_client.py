import unittest
from unittest.mock import MagicMock


def _client_returning(content: str) -> MagicMock:
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [MagicMock()]
    c.chat.completions.create.return_value.choices[0].message.content = content
    return c


class TestLLMClient(unittest.TestCase):
    def test_complete_text_returns_stripped_content(self):
        from src.llm.client import complete_text
        c = _client_returning('  {"a": 1}  ')
        self.assertEqual(complete_text("p", client=c), '{"a": 1}')

    def test_complete_json_parses(self):
        from src.llm.client import complete_json
        c = _client_returning('{"label": "x"}')
        self.assertEqual(complete_json("p", client=c), {"label": "x"})

    def test_complete_json_empty_and_bad_return_empty_dict(self):
        from src.llm.client import complete_json
        self.assertEqual(complete_json("p", client=_client_returning("")), {})
        self.assertEqual(complete_json("p", client=_client_returning("not json")), {})

    def test_complete_json_nonobject_returns_empty(self):
        from src.llm.client import complete_json
        self.assertEqual(complete_json("p", client=_client_returning("[1,2]")), {})

    def test_complete_vision_json_builds_image_message(self):
        from src.llm.client import complete_vision_json
        c = _client_returning('{"summary": "s"}')
        out = complete_vision_json(text="t", image_data_url="data:image/jpeg;base64,AAA", client=c)
        self.assertEqual(out, {"summary": "s"})
        msg = c.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(msg[0]["type"], "text")
        self.assertEqual(msg[1]["type"], "image_url")

    def test_temperature_zero_and_json_format(self):
        from src.llm.client import complete_text
        c = _client_returning("{}")
        complete_text("p", client=c)
        kw = c.chat.completions.create.call_args.kwargs
        self.assertEqual(kw["temperature"], 0.0)
        self.assertEqual(kw["response_format"], {"type": "json_object"})
