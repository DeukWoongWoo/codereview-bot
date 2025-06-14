import json  # For potential payload parsing if needed, though current bot.py does not use it in main
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

# Adjust path to import bot and chat
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    import gitlab  # Used for mocking gitlab client and exceptions

    from bot import GitlabBot
    from bot import main as bot_main
    from chat import Chat  # Used for mocking
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
        self.mock_gl_client_patch = patch("bot.gitlab.Gitlab")
        self.MockGitlabClientClass = self.mock_gl_client_patch.start()  # Start patch
        self.mock_gl_instance = (
            self.MockGitlabClientClass.return_value
        )  # Instance of Gitlab

        # Mock project and MR objects that the Gitlab client would return
        self.mock_project = MagicMock()
        self.mock_mr = MagicMock()
        self.mock_gl_instance.projects.get.return_value = self.mock_project
        self.mock_project.mergerequests.get.return_value = self.mock_mr

        # Mock Chat class that GitlabBot.load_chat() would use
        self.mock_chat_class_patch = patch("bot.Chat")
        self.MockChatClass = self.mock_chat_class_patch.start()
        self.mock_chat_instance = self.MockChatClass.return_value

        # Standard bot instance for most tests
        self.bot = GitlabBot(
            gitlab_url=self.fake_gitlab_url,
            gitlab_token=self.fake_gitlab_token,
            openai_api_key=self.fake_openai_key,
        )

    def tearDown(self):
        self.mock_gl_client_patch.stop()  # Stop patch
        self.mock_chat_class_patch.stop()  # Stop patch

    # --- Tests for _get_openai_params ---
    def test_get_openai_params_defaults(self):
        """Test _get_openai_params uses default values when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True): # Ensure clean env
            params = self.bot._get_openai_params()
        self.assertEqual(params["model"], "gpt-4o-mini")
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["top_p"], 1.0)
        self.assertIsNone(params["max_tokens_for_response"])

    @patch.dict(os.environ, {
        "MODEL": "test-model",
        "TEMPERATURE": "0.75",
        "TOP_P": "0.95",
        "MAX_TOKENS": "1024"
    })
    def test_get_openai_params_from_env(self):
        """Test _get_openai_params reads values from environment variables."""
        params = self.bot._get_openai_params()
        self.assertEqual(params["model"], "test-model")
        self.assertEqual(params["temperature"], 0.75)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(params["max_tokens_for_response"], 1024)

    @patch("bot.logger")
    @patch.dict(os.environ, {"TEMPERATURE": "invalid_float", "TOP_P": "another_invalid", "MAX_TOKENS": "not_an_int"})
    def test_get_openai_params_invalid_values_fallback_and_log(self, mock_logger):
        """Test _get_openai_params falls back to defaults and logs warnings for invalid values."""
        params = self.bot._get_openai_params()
        self.assertEqual(params["temperature"], 0.0) # Default
        self.assertEqual(params["top_p"], 1.0)       # Default
        self.assertIsNone(params["max_tokens_for_response"]) # Default (None)

        mock_logger.warning.assert_any_call("Invalid value for TEMPERATURE: 'invalid_float'. Using default 0.0.")
        mock_logger.warning.assert_any_call("Invalid value for TOP_P: 'another_invalid'. Using default 1.0.")
        mock_logger.warning.assert_any_call("Invalid value for MAX_TOKENS: 'not_an_int'. Using None.")

    # --- Tests for _get_mr_patch ---
    def test_get_mr_patch_success(self):
        """Test successful patch creation from MR changes."""
        mock_mr_obj = MagicMock(iid=101)
        mock_mr_obj.changes.return_value = {
            "changes": [
                {"new_path": "file1.py", "diff": "diff for file1"},
                {"new_path": "file2.py", "diff": "diff for file2"}
            ]
        }
        patch = self.bot._get_mr_patch(mock_mr_obj)
        expected_patch = "File: file1.py\ndiff for file1\n\nFile: file2.py\ndiff for file2\n\n"
        self.assertEqual(patch, expected_patch)

    @patch("bot.logger")
    def test_get_mr_patch_no_changes_key(self, mock_logger):
        """Test _get_mr_patch when 'changes' key is missing."""
        mock_mr_obj = MagicMock(iid=102)
        mock_mr_obj.changes.return_value = {} # No 'changes' key
        patch = self.bot._get_mr_patch(mock_mr_obj)
        self.assertIsNone(patch)
        mock_logger.error.assert_called_with("No changes found in merge request 102.")

    @patch("bot.logger")
    def test_get_mr_patch_empty_changes_list(self, mock_logger):
        """Test _get_mr_patch when 'changes' list is empty."""
        mock_mr_obj = MagicMock(iid=103)
        mock_mr_obj.changes.return_value = {"changes": []} # Empty list
        patch = self.bot._get_mr_patch(mock_mr_obj)
        self.assertIsNone(patch) # No diffs, so patch_parts will be empty
        mock_logger.info.assert_called_with("No diff content found in changes for MR 103.")


    @patch("bot.logger")
    def test_get_mr_patch_no_diff_content(self, mock_logger):
        """Test _get_mr_patch when changes are present but no 'diff' keys."""
        mock_mr_obj = MagicMock(iid=104)
        mock_mr_obj.changes.return_value = {
            "changes": [
                {"new_path": "file1.py"}, # No 'diff'
                {"new_path": "file2.py"}  # No 'diff'
            ]
        }
        patch = self.bot._get_mr_patch(mock_mr_obj)
        self.assertIsNone(patch) # patch_parts will be empty
        mock_logger.info.assert_called_with("No diff content found in changes for MR 104.")

    @patch("bot.logger")
    def test_get_mr_patch_gitlab_http_error(self, mock_logger):
        """Test _get_mr_patch handles GitlabHttpError during mr.changes()."""
        mock_mr_obj = MagicMock(iid=105)
        mock_mr_obj.changes.side_effect = gitlab.exceptions.GitlabHttpError("API error")
        patch = self.bot._get_mr_patch(mock_mr_obj)
        self.assertIsNone(patch)
        mock_logger.error.assert_called_with("GitLab API error while fetching changes for MR 105: API error")

    # --- Tests for _post_review_comment ---
    @patch("bot.logger")
    def test_post_review_comment_success(self, mock_logger):
        """Test successful posting of a review comment."""
        mock_mr_obj = MagicMock(iid=201)
        review_content = {"lgtm": True, "review_comment": "Looks great!"}

        self.bot._post_review_comment(mock_mr_obj, review_content)

        expected_body = "🤖 Code Review Bot\n\n✅ LGTM: True\n\nLooks great!"
        mock_mr_obj.notes.create.assert_called_once_with({'body': expected_body})
        mock_logger.info.assert_called_with("Successfully posted review comment to MR 201.")

    @patch("bot.logger")
    def test_post_review_comment_primary_fails_fallback_succeeds(self, mock_logger):
        """Test fallback comment posting if primary attempt fails."""
        mock_mr_obj = MagicMock(iid=202)
        review_content = {"lgtm": False, "review_comment": "Needs changes."}

        primary_error = gitlab.exceptions.GitlabCreateError("Primary post failed")
        mock_mr_obj.notes.create.side_effect = [primary_error, None] # Primary fails, fallback succeeds

        self.bot._post_review_comment(mock_mr_obj, review_content)

        expected_primary_body = "🤖 Code Review Bot\n\n⚠️ LGTM: False\n\nNeeds changes."
        expected_fallback_body = f"❌ An error occurred while trying to post the full review comment to MR 202.\nOriginal error: {primary_error}"

        calls = [
            call({'body': expected_primary_body}),
            call({'body': expected_fallback_body})
        ]
        mock_mr_obj.notes.create.assert_has_calls(calls)
        self.assertEqual(mock_mr_obj.notes.create.call_count, 2)
        mock_logger.error.assert_any_call(f"GitLab API error (GitlabCreateError) while posting review comment to MR 202: {primary_error}")
        mock_logger.info.assert_any_call("Posted simplified error notification to MR 202.")

    @patch("bot.logger")
    def test_post_review_comment_primary_and_fallback_fail(self, mock_logger):
        """Test failure of both primary and fallback comment posting."""
        mock_mr_obj = MagicMock(iid=203)
        review_content = {"lgtm": True, "review_comment": "Excellent!"}

        primary_error = gitlab.exceptions.GitlabCreateError("Primary post failed")
        fallback_error = gitlab.exceptions.GitlabCreateError("Fallback post also failed")
        mock_mr_obj.notes.create.side_effect = [primary_error, fallback_error]

        self.bot._post_review_comment(mock_mr_obj, review_content)

        self.assertEqual(mock_mr_obj.notes.create.call_count, 2)
        mock_logger.error.assert_any_call(f"GitLab API error (GitlabCreateError) while posting review comment to MR 203: {primary_error}")
        mock_logger.error.assert_any_call(f"Failed to post even the simplified error notification to MR 203: {fallback_error}", exc_info=True)

    @patch("bot.logger")
    def test_post_review_comment_unexpected_error(self, mock_logger):
        """Test handling of an unexpected error during comment formatting/posting."""
        mock_mr_obj = MagicMock(iid=204)
        review_content = {"lgtm": True, "review_comment": "Superb!"}

        unexpected_error = Exception("Something broke unexpectedly")
        mock_mr_obj.notes.create.side_effect = unexpected_error

        self.bot._post_review_comment(mock_mr_obj, review_content)

        mock_mr_obj.notes.create.assert_called_once() # Initial attempt
        # Second attempt (critical error message) might also fail or succeed depending on mock setup for it.
        # For this test, we're mainly interested in the logging of the unexpected error.
        mock_logger.error.assert_any_call(f"Unexpected error while formatting or posting review comment to MR 204: {unexpected_error}", exc_info=True)


    # --- Tests for handle_merge_request orchestration ---

    @patch("bot.GitlabBot._post_review_comment")
    @patch("bot.GitlabBot._get_mr_patch")
    @patch("bot.GitlabBot._get_openai_params")
    @patch("bot.GitlabBot.load_chat")
    @patch("bot.logger")
    def test_handle_merge_request_success_orchestration(
        self, mock_logger, mock_load_chat, mock_get_openai_params,
        mock_get_mr_patch, mock_post_review_comment
    ):
        """Test successful orchestration by handle_merge_request."""
        # Setup mocks for helper methods
        # self.mock_chat_instance is from the main TestGitlabBot setUp
        mock_load_chat.return_value = self.mock_chat_instance
        mock_openai_params_dict = {
            "model": "test-model", "temperature": 0.1, "top_p": 0.9, "max_tokens_for_response": 500
        }
        mock_get_openai_params.return_value = mock_openai_params_dict
        mock_get_mr_patch.return_value = "fake patch content"

        # Configure the mock chat instance (which is returned by mock_load_chat)
        self.mock_chat_instance.code_review.return_value = SAMPLE_SUCCESS_REVIEW

        # self.mock_mr is set up in TestGitlabBot.setUp
        self.bot.handle_merge_request(project_id=123, merge_request_iid=789)

        self.mock_gl_instance.projects.get.assert_called_once_with(123)
        self.mock_project.mergerequests.get.assert_called_once_with(789)
        mock_load_chat.assert_called_once()
        mock_get_mr_patch.assert_called_once_with(self.mock_mr)
        mock_get_openai_params.assert_called_once()

        self.mock_chat_instance.code_review.assert_called_once_with(
            patch="fake patch content",
            model=mock_openai_params_dict["model"],
            temperature=mock_openai_params_dict["temperature"],
            top_p=mock_openai_params_dict["top_p"],
            max_tokens_for_response=mock_openai_params_dict["max_tokens_for_response"]
        )
        mock_post_review_comment.assert_called_once_with(self.mock_mr, SAMPLE_SUCCESS_REVIEW)
        mock_logger.error.assert_not_called()

    @patch("bot.logger")
    def test_handle_merge_request_gitlab_get_error(self, mock_logger):
        """Test handle_merge_request when GitlabGetError occurs during MR retrieval."""
        self.mock_gl_instance.projects.get.side_effect = gitlab.exceptions.GitlabGetError("Failed to get project")

        self.bot.handle_merge_request(project_id=999, merge_request_iid=1)

        mock_logger.error.assert_any_call(
            "GitLab API error (GitlabGetError) while retrieving project or MR: Failed to get project. ProjectID: 999, MR_IID: 1."
        )
        # Ensure other methods like load_chat are not called (via checking code_review on the chat instance)
        self.mock_chat_instance.code_review.assert_not_called()


    @patch("bot.GitlabBot._get_mr_patch", return_value=None) # Mock _get_mr_patch to return None
    @patch("bot.logger")
    def test_handle_merge_request_no_patch_halts_processing(self, mock_logger, mock_get_mr_patch_none):
        """Test that review halts if _get_mr_patch returns None."""
        # self.mock_mr is available from setUp, and project/MR retrieval will be mocked to succeed
        self.bot.handle_merge_request(project_id=123, merge_request_iid=789)

        mock_get_mr_patch_none.assert_called_once_with(self.mock_mr)
        self.mock_chat_instance.code_review.assert_not_called()
        # _get_mr_patch logs its own errors, so no specific error log to check here from handle_merge_request directly.

    @patch("bot.GitlabBot._post_review_comment")
    @patch("bot.GitlabBot._get_mr_patch", return_value="fake patch content") # Assume patch is fine
    @patch("bot.GitlabBot._get_openai_params")
    @patch("bot.GitlabBot.load_chat")
    @patch("bot.logger")
    def test_handle_merge_request_code_review_exception_posts_error(
        self, mock_logger, mock_load_chat, mock_get_openai_params,
        mock_get_mr_patch, mock_post_review_comment
    ):
        """Test that an error comment is posted if chat.code_review fails."""
        mock_load_chat.return_value = self.mock_chat_instance
        mock_get_openai_params.return_value = {"model": "test", "temperature": 0.1, "top_p": 1.0, "max_tokens_for_response": 100}

        review_exception = Exception("AI go boom")
        self.mock_chat_instance.code_review.side_effect = review_exception

        self.bot.handle_merge_request(project_id=123, merge_request_iid=789)

        self.mock_chat_instance.code_review.assert_called_once()
        mock_logger.error.assert_any_call(f"Code review generation failed for MR {self.mock_mr.iid}: {review_exception}", exc_info=True)

        expected_error_review = {
            "lgtm": False,
            "review_comment": f"❌ Code review generation failed due to an internal error: {review_exception}"
        }
        mock_post_review_comment.assert_called_once_with(self.mock_mr, expected_error_review)

    @patch("bot.logger")
    def test_handle_merge_request_chat_init_key_missing(self, mock_logger):
        # Create bot with no OpenAI key
        bot_no_key = GitlabBot(
            self.fake_gitlab_url, self.fake_gitlab_token, openai_api_key=""
        )

        bot_no_key.handle_merge_request(123, 789)

        mock_logger.error.assert_any_call("OPENAI_API_KEY is not set")  # From load_chat
        mock_logger.error.assert_any_call(
            "Chat initialization failed"
        )  # From handle_merge_request
        self.mock_mr.changes.assert_not_called()  # Should exit before processing MR

    @patch("bot.logger")
    def test_handle_merge_request_chat_init_exception(self, mock_logger):
        self.MockChatClass.side_effect = Exception("Chat Boom!")  # Chat init fails

        # Bot creation is fine, error happens in load_chat
        self.bot.handle_merge_request(123, 789)

        self.MockChatClass.assert_called_once_with(self.fake_openai_key)
        mock_logger.error.assert_any_call(
            "Failed to initialize Chat: Chat Boom!"
        )  # From load_chat
        mock_logger.error.assert_any_call(
            "Chat initialization failed"
        )  # From handle_merge_request
        self.mock_mr.changes.assert_not_called()

    # Removed: test_handle_merge_request_no_changes (covered by _get_mr_patch tests and no_patch_halts_processing)
    # Removed: test_handle_merge_request_code_review_error (covered by code_review_exception_posts_error)
    # Removed: test_handle_merge_request_gitlab_comment_error (covered by _post_review_comment tests)

class TestMainFunction(unittest.TestCase):

    @patch.dict(
        os.environ,
        {
            "CI_SERVER_URL": "http://fakegitlab.com",
            "CI_PROJECT_ID": "100",
            "CI_MERGE_REQUEST_IID": "200",
            "GITLAB_TOKEN": "main_fake_token",
            "OPENAI_API_KEY": "main_fake_openai_key",
        },
    )
    @patch("bot.GitlabBot")  # Mock the GitlabBot class in bot.py
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
            "OPENAI_API_KEY": "key",
        }

        for var_to_omit in required_vars_map.keys():
            current_env = {
                k: v for k, v in required_vars_map.items() if k != var_to_omit
            }

            with patch.dict(os.environ, current_env, clear=True):
                with self.assertRaises(ValueError) as context:
                    bot_main()
                self.assertTrue(
                    f"Missing required environment variables: {var_to_omit}"
                    in str(context.exception)
                    or f"Missing required environment variables: {var_to_omit.lower()}"
                    in str(
                        context.exception
                    )  # some keys might be lowercased by getenv by mistake
                    or var_to_omit in str(context.exception)
                )  # General check


if __name__ == "__main__":
    unittest.main()
