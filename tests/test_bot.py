import sys
import types
import unittest
from unittest.mock import MagicMock
# ruff: noqa: E402

sys.modules['gitlab'] = types.ModuleType('gitlab')
sys.modules['gitlab'].Gitlab = MagicMock()
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MagicMock()
sys.modules['mlflow'] = types.ModuleType('mlflow')
sys.modules['mlflow'].load_prompt = MagicMock()
sys.modules['yaml'] = types.ModuleType('yaml')
sys.modules['yaml'].safe_load = MagicMock(return_value={})

from bot import GitlabBot


class BotTests(unittest.TestCase):
    def test_load_chat_missing_key(self):
        bot = GitlabBot('url', 'token', '')
        self.assertIsNone(bot.load_chat())

    def test_handle_merge_request(self):
        bot = GitlabBot('url', 'token', 'key')
        bot.gl = MagicMock()
        project = MagicMock()
        mr = MagicMock()
        bot.gl.projects.get.return_value = project
        project.mergerequests.get.return_value = mr

        mr.changes.return_value = {'changes': [{'new_path': 'f', 'diff': 'd'}]}
        mr.notes.list.return_value = []
        mr.description = 'desc'
        mr.created_at = 'now'
        mr.notes.create = MagicMock()

        bot.load_chat = MagicMock()
        bot.load_chat.return_value = MagicMock(
            code_review_agent=MagicMock(return_value={'lgtm': True, 'review_comment': 'ok'})
        )

        bot.handle_merge_request(1, 1)
        self.assertTrue(mr.notes.create.called)


if __name__ == '__main__':
    unittest.main()
