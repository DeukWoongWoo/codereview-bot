import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import json # For potential payload parsing if needed, though current bot.py does not use it in main

# Adjust path to import bot and chat
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from bot import GitlabBot, main as bot_main
    from chat import Chat # Used for mocking
    import gitlab # Used for mocking gitlab client and exceptions
except ImportError as e:
    print(f"Error importing modules for test_bot.py: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

# Sample successful code review from Chat
SAMPLE_SUCCESS_REVIEW = {"lgtm": True, "review_comment": "Looks good!"}
SAMPLE_FAIL_REVIEW = {"lgtm": False, "review_comment": "Needs fixing."}

class TestGitlabBot(unittest.TestCase):

    def setUp(self):
        self.fake_gitlab_url = "http://fakegitlab.com"
        self.fake_gitlab_token = "fake_token"
        self.fake_openai_key = "fake_openai_key"

        # Mock the gitlab.Gitlab client
        self.mock_gl_client_patch = patch('bot.gitlab.Gitlab')
        self.MockGitlabClientClass = self.mock_gl_client_patch.start() # Start patch
        self.mock_gl_instance = self.MockGitlabClientClass.return_value # Instance of Gitlab

        # Mock project and MR objects that the Gitlab client would return
        self.mock_project = MagicMock()
        self.mock_mr = MagicMock()
        self.mock_gl_instance.projects.get.return_value = self.mock_project
        self.mock_project.mergerequests.get.return_value = self.mock_mr

        # Mock Chat class that GitlabBot.load_chat() would use
        self.mock_chat_class_patch = patch('bot.Chat')
        self.MockChatClass = self.mock_chat_class_patch.start()
        self.mock_chat_instance = self.MockChatClass.return_value

        # Standard bot instance for most tests
        self.bot = GitlabBot(
            gitlab_url=self.fake_gitlab_url,
            gitlab_token=self.fake_gitlab_token,
            openai_api_key=self.fake_openai_key
        )

    def tearDown(self):
        self.mock_gl_client_patch.stop() # Stop patch
        self.mock_chat_class_patch.stop() # Stop patch

    @patch('bot.logger') # Mock logger within bot.py
    def test_handle_merge_request_success(self, mock_logger):
        # Configure MR data
        self.mock_mr.description = "Test MR"
        self.mock_mr.created_at = "2023-01-01T00:00:00Z"
        self.mock_mr.changes.return_value = {
            "changes": [{"new_path": "file.py", "diff": "fake diff content"}]
        }
        # Configure Chat mock
        self.mock_chat_instance.code_review.return_value = SAMPLE_SUCCESS_REVIEW

        # Call the method
        self.bot.handle_merge_request(project_id=123, merge_request_iid=789)

        # Assertions
        self.mock_gl_instance.projects.get.assert_called_once_with(123)
        self.mock_project.mergerequests.get.assert_called_once_with(789)
        self.MockChatClass.assert_called_once_with(self.fake_openai_key) # Chat init

        expected_patch_arg = "File: file.py\nfake diff content\n\n"
        expected_code_review_input = {
            "description": "Test MR",
            "patch": expected_patch_arg,
            "created_at": "2023-01-01T00:00:00Z"
        }
        # Get env var defaults for model, temp, etc. to match what bot.py uses
        expected_model = os.getenv('MODEL', 'gpt-4o-mini')
        expected_temp = float(os.getenv('TEMPERATURE', 0.0))
        expected_top_p = float(os.getenv('TOP_P', 1.0))
        expected_max_tokens = os.getenv('MAX_TOKENS', None)

        self.mock_chat_instance.code_review.assert_called_once_with(
            expected_code_review_input, expected_model, expected_temp, expected_top_p, expected_max_tokens
        )

        expected_comment_body = (
            "🤖 Code Review Bot\n\n"
            f"✅ LGTM: True\n\n"
            f"{SAMPLE_SUCCESS_REVIEW['review_comment']}"
        )
        self.mock_mr.notes.create.assert_called_once_with({'body': expected_comment_body})
        mock_logger.error.assert_not_called() # No errors logged

    @patch('bot.logger')
    def test_handle_merge_request_chat_init_key_missing(self, mock_logger):
        # Create bot with no OpenAI key
        bot_no_key = GitlabBot(self.fake_gitlab_url, self.fake_gitlab_token, openai_api_key="")

        bot_no_key.handle_merge_request(123, 789)

        mock_logger.error.assert_any_call("OPENAI_API_KEY is not set") # From load_chat
        mock_logger.error.assert_any_call("Chat initialization failed") # From handle_merge_request
        self.mock_mr.changes.assert_not_called() # Should exit before processing MR

    @patch('bot.logger')
    def test_handle_merge_request_chat_init_exception(self, mock_logger):
        self.MockChatClass.side_effect = Exception("Chat Boom!") # Chat init fails

        # Bot creation is fine, error happens in load_chat
        self.bot.handle_merge_request(123, 789)

        self.MockChatClass.assert_called_once_with(self.fake_openai_key)
        mock_logger.error.assert_any_call("Failed to initialize Chat: Chat Boom!") # From load_chat
        mock_logger.error.assert_any_call("Chat initialization failed") # From handle_merge_request
        self.mock_mr.changes.assert_not_called()

    @patch('bot.logger')
    def test_handle_merge_request_no_changes(self, mock_logger):
        self.mock_mr.changes.return_value = {"changes": []} # No changes in MR

        self.bot.handle_merge_request(123, 789)

        mock_logger.error.assert_called_once_with("No changes found in merge request")
        self.mock_chat_instance.code_review.assert_not_called()

    @patch('bot.logger')
    def test_handle_merge_request_code_review_error(self, mock_logger):
        self.mock_mr.changes.return_value = {"changes": [{"new_path": "f.py", "diff": "d"}]}
        self.mock_mr.description = "desc"
        self.mock_mr.created_at = "time"
        self.mock_chat_instance.code_review.side_effect = Exception("OpenAI Boom!")

        self.bot.handle_merge_request(123, 789)

        self.mock_chat_instance.code_review.assert_called_once()
        mock_logger.error.assert_called_once_with("Error during code review: OpenAI Boom!")
        self.mock_mr.notes.create.assert_not_called()

    @patch('bot.logger')
    def test_handle_merge_request_gitlab_comment_error(self, mock_logger):
        self.mock_mr.changes.return_value = {"changes": [{"new_path": "f.py", "diff": "d"}]}
        self.mock_mr.description = "desc"
        self.mock_mr.created_at = "time"
        self.mock_chat_instance.code_review.return_value = SAMPLE_FAIL_REVIEW

        # First call (actual review) fails, second call (error message) succeeds
        self.mock_mr.notes.create.side_effect = [
            gitlab.exceptions.GitlabCreateError("Post failed"),
            None
        ]

        self.bot.handle_merge_request(123, 789)

        self.mock_chat_instance.code_review.assert_called_once()

        expected_fail_comment_body = (
            "🤖 Code Review Bot\n\n"
            f"⚠️ LGTM: False\n\n"
            f"{SAMPLE_FAIL_REVIEW['review_comment']}"
        )
        expected_error_comment_body = (
            f"❌ An error occurred while reviewing this merge request.\n"
            f" error message: Post failed\n\n"
            f" original review comment : {SAMPLE_FAIL_REVIEW['review_comment']}"
        )

        self.mock_mr.notes.create.assert_has_calls([
            call({'body': expected_fail_comment_body}),
            call({'body': expected_error_comment_body})
        ])
        # This logger message is from the first exception block in notes.create
        mock_logger.error.assert_called_once_with("Error during code review: Post failed")


class TestMainFunction(unittest.TestCase):

    @patch.dict(os.environ, {
        "CI_SERVER_URL": "http://fakegitlab.com",
        "CI_PROJECT_ID": "100",
        "CI_MERGE_REQUEST_IID": "200",
        "GITLAB_TOKEN": "main_fake_token",
        "OPENAI_API_KEY": "main_fake_openai_key"
    })
    @patch('bot.GitlabBot') # Mock the GitlabBot class in bot.py
    def test_main_success(self, MockGitlabBotClass):
        mock_bot_instance = MockGitlabBotClass.return_value

        bot_main()

        MockGitlabBotClass.assert_called_once_with(
            "http://fakegitlab.com", "main_fake_token", "main_fake_openai_key"
        )
        mock_bot_instance.handle_merge_request.assert_called_once_with("100", "200")

    def test_main_missing_env_vars(self):
        required_vars_map = {
            "CI_SERVER_URL": "http://fake.com",
            "CI_PROJECT_ID": "1",
            "CI_MERGE_REQUEST_IID": "2",
            "GITLAB_TOKEN": "token",
            "OPENAI_API_KEY": "key"
        }

        for var_to_omit in required_vars_map.keys():
            current_env = {k: v for k, v in required_vars_map.items() if k != var_to_omit}

            with patch.dict(os.environ, current_env, clear=True):
                with self.assertRaises(ValueError) as context:
                    bot_main()
                self.assertTrue(f"Missing required environment variables: {var_to_omit}" in str(context.exception) or
                                f"Missing required environment variables: {var_to_omit.lower()}" in str(context.exception) or # some keys might be lowercased by getenv by mistake
                                var_to_omit in str(context.exception)) # General check


if __name__ == '__main__':
    unittest.main()
