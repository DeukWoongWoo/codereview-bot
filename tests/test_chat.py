import sys
import types
import unittest
from unittest.mock import patch, MagicMock
# ruff: noqa: E402

sys.modules['mlflow'] = types.ModuleType('mlflow')
sys.modules['mlflow'].load_prompt = MagicMock()
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MagicMock()
sys.modules['yaml'] = types.ModuleType('yaml')
sys.modules['yaml'].safe_load = MagicMock(return_value={})

from chat import Chat


class ChatTests(unittest.TestCase):
    @patch('chat.OpenAI')
    def test_generate_prompt(self, mock_openai):
        chat = Chat('key')
        ctx = {'description': 'desc', 'comments': 'c', 'patch': 'p'}
        prompt = chat._generate_prompt(ctx)
        self.assertIn('desc', prompt)
        self.assertIn('Patch', prompt)

    @patch('chat.OpenAI')
    def test_code_review_agent_split(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        chat = Chat('key')
        chat.assistant_id = 'aid'
        chat._assistant_review = MagicMock(side_effect=[{'lgtm': True, 'review_comment': 'a'}, {'lgtm': False, 'review_comment': 'b'}])
        ctx = {'description': '', 'comments': '', 'patch': 'a\n'*4}
        result = chat.code_review_agent(ctx, chunk_size=5)
        self.assertFalse(result['lgtm'])
        self.assertIn('a', result['review_comment'])
        self.assertIn('b', result['review_comment'])
        self.assertEqual(chat._assistant_review.call_count, 2)


if __name__ == '__main__':
    unittest.main()
