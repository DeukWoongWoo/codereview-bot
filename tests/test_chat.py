import unittest
from unittest.mock import patch, mock_open, MagicMock
import yaml
import os
import json

# It's good practice to import the module under test after setting up potential mocks if needed,
# or ensure mocks are applied correctly. Here, we're importing Chat from the 'chat' module.
# Assuming chat.py is in the root directory, and tests are in tests/
# We might need to adjust sys.path if running tests directly and chat.py isn't found.
# However, typical test runners (like pytest or unittest discovery) handle this.
try:
    from chat import Chat # Assuming chat.py is in the parent directory or PYTHONPATH
except ImportError:
    # Adjust path for local testing if necessary
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from chat import Chat, ACTUAL_DEFAULT_PROMPTS as CHAT_ACTUAL_DEFAULT_PROMPTS # Import for comparison
    # Make pandas an optional import for tests if chat.py treats it as such
    try:
        import pandas as pd
    except ImportError:
        pd = None # Mock or skip pandas-dependent tests if not available
except ImportError:
    # Adjust path for local testing if necessary
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from chat import Chat, ACTUAL_DEFAULT_PROMPTS as CHAT_ACTUAL_DEFAULT_PROMPTS
    try:
        import pandas as pd
    except ImportError:
        pd = None

from openai import OpenAIError
# For mocking MLflow specific exceptions
try:
    from mlflow.exceptions import MlflowException
except ImportError:
    # Define a dummy exception if mlflow is not installed in the test environment
    # This allows tests to run and be skipped if mlflow components are missing
    class MlflowException(Exception):
        pass


# This is the actual default structure used in chat.py's _load_prompts
# Re-aliasing after potential import from chat.py to ensure it's the one from this file for clarity
# if the one from chat.py is not imported or named differently.
# However, it's better to use the one imported from chat.py for direct comparison.
ACTUAL_DEFAULT_PROMPTS_IN_TEST_FILE_FOR_REFERENCE_ONLY = { # Renamed to avoid confusion
    "default_review_prompt": "Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.",
    "json_format_requirement": "Provide your feedback in a strict JSON format with the following structure:\n{\n    \"lgtm\": boolean, // true if the code looks good to merge, false if there are concerns\n    \"review_comment\": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object."
}

SAMPLE_YAML_PROMPTS = {
    "default_review_prompt": "Custom review prompt from YAML.",
    "json_format_requirement": "Custom JSON format from YAML."
}
SAMPLE_YAML_CONTENT = yaml.dump(SAMPLE_YAML_PROMPTS)

# Prompts for MLflow testing
MLFLOW_PROMPTS = {
    "default_review_prompt": "MLflow artifact prompt.",
    "json_format_requirement": "MLflow JSON format."
}
MLFLOW_PROMPTS_YAML_CONTENT = yaml.dump(MLFLOW_PROMPTS)

# Test class for general Chat functionalities (excluding specific _load_prompts and chunking)
class TestChatBase(unittest.TestCase):

    @patch('chat.Chat._load_prompts')
    def setUp(self, mock_load_prompts):
        mock_load_prompts.return_value = SAMPLE_YAML_PROMPTS # Default for these tests
        self.chat = Chat(api_key="test_api_key")
        self.chat.openai = MagicMock() # Mock OpenAI client

    def test_generate_prompt_default(self):
        # Use CHAT_ACTUAL_DEFAULT_PROMPTS which is imported from chat.py
        self.chat.prompts = CHAT_ACTUAL_DEFAULT_PROMPTS
        patch_content = "test patch content"
        # Check against the structure _generate_prompt now creates
        expected_prompt_string = f"{CHAT_ACTUAL_DEFAULT_PROMPTS['default_review_prompt']}\n{CHAT_ACTUAL_DEFAULT_PROMPTS['json_format_requirement']}\n--- BEGIN PATCH ---\n{patch_content}\n--- END PATCH ---"

        with patch.dict(os.environ, {}, clear=True):
             actual_prompt_string = self.chat._generate_prompt(patch_content)
        self.assertEqual(actual_prompt_string, expected_prompt_string)

    @patch.dict(os.environ, {"PROMPT": "Custom prompt from ENV VAR"})
    def test_generate_prompt_with_env_var(self):
        self.chat.prompts = CHAT_ACTUAL_DEFAULT_PROMPTS
        patch_content = "test patch content"
        env_prompt = "Custom prompt from ENV VAR"
        expected_prompt_string = f"{env_prompt}\n{CHAT_ACTUAL_DEFAULT_PROMPTS['json_format_requirement']}\n--- BEGIN PATCH ---\n{patch_content}\n--- END PATCH ---"

        actual_prompt_string = self.chat._generate_prompt(patch_content)
        self.assertEqual(actual_prompt_string, expected_prompt_string)

    # Test for code_review (non-chunking aspects) can remain here or move to a dedicated class
    # For now, keeping simple success/error cases here. Chunking tests will be separate.
    @patch('chat.MODEL_TOKEN_LIMITS', {'gpt-test': 10000}) # Ensure no chunking for this test
    @patch('chat.Chat._get_token_count', return_value=100) # Mock token count to be low
    def test_code_review_success_no_chunking(self, mock_get_tokens, mock_model_limits):
        self.chat.prompts = SAMPLE_YAML_PROMPTS # Ensure prompts are loaded for _generate_prompt
        patch_content = "short patch"
        expected_review = {"lgtm": True, "review_comment": "Looks good!"}

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(expected_review)))]
        self.chat.openai.chat.completions.create.return_value = mock_completion

        with patch('chat.logger') as mock_logger: # Mock logger to check timing log
            review = self.chat.code_review(patch_content, model="gpt-test")

        self.assertEqual(review, expected_review)
        self.chat.openai.chat.completions.create.assert_called_once()
        # Check that the logger.info about single chunk was called
        self.assertTrue(any("Processing as a single request" in call_args[0][0] for call_args in mock_logger.info.call_args_list))


# Separate class for testing the _load_prompts logic with MLflow and fallbacks
class TestChatPromptLoading(unittest.TestCase):

    def setUp(self):
        # These mocks will be started and stopped for each test method via decorators or context managers
        self.mock_mlflow_set_uri = patch('chat.mlflow.set_tracking_uri').start()
        # Removed: self.mock_mlflow_download = patch('chat.mlflow.artifacts.download_artifacts').start()
        # Removed: self.mock_mlflow_load_model = patch('chat.mlflow.pyfunc.load_model').start()
        self.mock_mlflow_load_prompt = patch('chat.mlflow.prompts.load_prompt').start() # Updated mock target

        self.mock_os_path_exists = patch('os.path.exists').start()
        self.mock_open_builtin = patch('builtins.open', new_callable=mock_open).start()

        # Mock logger to check log messages
        self.mock_logger = patch('chat.logger').start()

        # Ensure pandas is available for model loading tests, or skip them
        if pd is None:
            self.skip_pandas_tests = True # Flag to skip model loading tests
        else:
            self.skip_pandas_tests = False


    def tearDown(self):
        patch.stopall() # Stops all patches started with patch().start()

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_NAME": "test_prompt"
        # No version - tests default fetching
    })
    def test_load_prompts_mlflow_registry_success_no_version(self):
        self.mock_mlflow_load_prompt.return_value = MLFLOW_PROMPTS_YAML_CONTENT

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, MLFLOW_PROMPTS)
        self.mock_mlflow_set_uri.assert_called_once_with("http://fake-mlflow-uri")
        self.mock_mlflow_load_prompt.assert_called_once_with(uri="prompts:/test_prompt")
        self.mock_logger.info.assert_any_call("Successfully loaded and validated prompts from MLflow Prompt Registry.")
        self.mock_os_path_exists.assert_not_called()

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_NAME": "test_prompt_versioned",
        "MLFLOW_PROMPT_VERSION": "2"
    })
    def test_load_prompts_mlflow_registry_success_with_version(self):
        self.mock_mlflow_load_prompt.return_value = MLFLOW_PROMPTS_YAML_CONTENT

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, MLFLOW_PROMPTS)
        self.mock_mlflow_set_uri.assert_called_once_with("http://fake-mlflow-uri")
        self.mock_mlflow_load_prompt.assert_called_once_with(uri="prompts:/test_prompt_versioned/2")
        self.mock_logger.info.assert_any_call("Successfully loaded and validated prompts from MLflow Prompt Registry.")

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_NAME": "test_prompt_fail"
    })
    def test_load_prompts_mlflow_registry_failure_fallback_to_yaml(self):
        self.mock_mlflow_load_prompt.side_effect = MlflowException("Registry error")
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_logger.error.assert_any_call("MlflowException occurred while loading prompts from Prompt Registry: Registry error")
        self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_NAME": "test_prompt_invalid_yaml"
    })
    def test_load_prompts_mlflow_registry_invalid_yaml_fallback_to_yaml(self):
        self.mock_mlflow_load_prompt.return_value = "this: is: invalid: yaml"
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

        chat_instance = Chat(api_key="key")
        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_logger.error.assert_any_call("Error parsing YAML from MLflow Prompt Registry: " + str(yaml.YAMLError("dummy error message"))) # Check type of error
        self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_NAME": "test_prompt_invalid_structure"
    })
    def test_load_prompts_mlflow_registry_invalid_structure_fallback_to_yaml(self):
        invalid_structure_prompts = {"only_one_key": "value"}
        self.mock_mlflow_load_prompt.return_value = yaml.dump(invalid_structure_prompts)
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

        chat_instance = Chat(api_key="key")
        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_logger.warning.assert_any_call("Prompts loaded from MLflow Prompt Registry are invalid or missing expected keys. Proceeding to fallbacks.")
        self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")


    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://fake-mlflow-uri"}) # MLFLOW_PROMPT_NAME is missing
    def test_load_prompts_mlflow_not_fully_configured_fallback_to_yaml(self):
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_mlflow_set_uri.assert_not_called()
        self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")

    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
        "MLFLOW_PROMPT_SOURCE": "artifact",
         # Missing RUN_ID and ARTIFACT_PATH intentionally
    })
    # This test was for the old artifact loading logic and is no longer relevant
    # def test_load_prompts_mlflow_artifact_misconfigured_fallback_to_yaml(self):
    #     self.mock_os_path_exists.return_value = True # prompts.yaml exists
    #     self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

    #     chat_instance = Chat(api_key="key")
    #     self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
    #     self.mock_logger.warning.assert_any_call("MLFLOW_RUN_ID and MLFLOW_ARTIFACT_PATH must be set for source 'artifact'.")
    #     self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")


    @patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri", # Configured for MLflow
        "MLFLOW_PROMPT_NAME": "test_prompt_hardcoded_fallback"
    })
    def test_load_prompts_all_fallbacks_to_hardcoded(self):
        self.mock_mlflow_load_prompt.side_effect = MlflowException("Registry error") # Corrected to use the new mock
        self.mock_os_path_exists.return_value = False # prompts.yaml does not exist
        # mock_open will not be called for local prompts.yaml if os.path.exists is false

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, CHAT_ACTUAL_DEFAULT_PROMPTS)
        self.mock_logger.error.assert_any_call("MLflowException occurred while loading prompts: Artifact download failed")
        self.mock_logger.info.assert_any_call("Local prompts.yaml not found.")
        self.mock_logger.warning.assert_any_call("Falling back to hardcoded default prompts.")


# Placeholder for chunking tests - to be implemented next
class TestChatCodeReviewChunking(unittest.TestCase):
    def setUp(self):
        # Mock Chat's _load_prompts to return some default prompts for these tests
        # This avoids dealing with MLflow/file loading here.
        self.mock_load_prompts_patch = patch('chat.Chat._load_prompts')
        self.MockLoadPrompts = self.mock_load_prompts_patch.start()
        self.MockLoadPrompts.return_value = CHAT_ACTUAL_DEFAULT_PROMPTS

        self.chat = Chat(api_key="test_api_key")
        self.chat.openai = MagicMock() # Mock OpenAI client

         # Mock tiktoken
        self.mock_tiktoken_encoding_for_model = patch('chat.tiktoken.encoding_for_model').start()
        self.mock_encoder = MagicMock()
        self.mock_tiktoken_encoding_for_model.return_value = self.mock_encoder
        # Default behavior for encode: return list with length of string / 4 (approx)
        self.mock_encoder.encode.side_effect = lambda text: list(range(len(text) // 4))


    def tearDown(self):
        patch.stopall()

    def test_get_token_count_known_model(self):
        self.mock_encoder.encode.return_value = [1, 2, 3, 4, 5] # Simulate 5 tokens
        count = self.chat._get_token_count("some text", "gpt-4o-mini")
        self.assertEqual(count, 5)
        self.mock_tiktoken_encoding_for_model.assert_called_once_with("gpt-4o-mini")
        self.mock_encoder.encode.assert_called_once_with("some text")

    @patch('chat.tiktoken.get_encoding') # For fallback
    def test_get_token_count_unknown_model_fallback(self, mock_get_encoding):
        self.mock_tiktoken_encoding_for_model.side_effect = KeyError("Model not found")
        mock_default_encoder = MagicMock()
        mock_default_encoder.encode.return_value = [1,2,3]
        mock_get_encoding.return_value = mock_default_encoder

        with patch('chat.logger') as mock_logger:
            count = self.chat._get_token_count("text", "unknown-model")

        self.assertEqual(count, 3)
        mock_get_encoding.assert_called_once_with("cl100k_base") # chat.DEFAULT_ENCODER
        mock_logger.warning.assert_called_once()

    def test_split_patch_no_split_needed(self):
        # Mock _get_token_count to return a low number so no splitting occurs
        self.chat._get_token_count = MagicMock(return_value=10)
        patch_content = "diff --git a/file1.py b/file1.py\n--- a/file1.py\n+++ b/file1.py\n@@ -1,1 +1,1 @@\n-old\n+new"
        chunks = self.chat._split_patch(patch_content, max_tokens_per_chunk=100, model="gpt-test")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], patch_content)

    def test_split_patch_by_file(self):
        patch_content = (
            "diff --git a/file1.py b/file1.py\n--- a/file1.py\n+++ b/file1.py\n@@ -1,1 +1,1 @@\n-f1 old\n+f1 new\n"
            "diff --git a/file2.py b/file2.py\n--- a/file2.py\n+++ b/file2.py\n@@ -1,1 +1,1 @@\n-f2 old\n+f2 new"
        )
        # Mock _get_token_count: first call for file1 (fits), second for file2 (fits)
        self.chat._get_token_count = MagicMock(side_effect=[50, 50])
        chunks = self.chat._split_patch(patch_content, max_tokens_per_chunk=100, model="gpt-test")
        self.assertEqual(len(chunks), 2)
        self.assertTrue("file1.py" in chunks[0])
        self.assertTrue("file2.py" in chunks[1])

    def test_split_patch_large_file_needs_line_splitting(self):
        file1_header = "diff --git a/file1.py b/file1.py\n--- a/file1.py\n+++ b/file1.py\n"
        file1_line1 = "@@ -1,1 +1,1 @@\n-old line1\n+new line1\n" # assume 10 tokens
        file1_line2 = "@@ -10,1 +10,1 @@\n-old line2\n+new line2\n" # assume 10 tokens
        patch_content = file1_header + file1_line1 + file1_line2

        # Token counts: file_diff (header+l1+l2) > max_chunk. Then lines.
        # Header tokens (approx, as _split_patch calculates it from joined header lines)
        header_tokens = self._calculate_mock_tokens(file1_header, self.mock_encoder.encode)
        line1_tokens = self._calculate_mock_tokens(file1_line1, self.mock_encoder.encode)
        line2_tokens = self._calculate_mock_tokens(file1_line2, self.mock_encoder.encode)

        # Overall diff, then individual lines/headers for splitting logic
        self.chat._get_token_count = MagicMock(side_effect=[
            header_tokens + line1_tokens + line2_tokens, # Full file1_diff
            header_tokens, # Header for first sub-chunk
            line1_tokens,  # Line1 for first sub-chunk
            header_tokens, # Header for second sub-chunk (after split)
            line2_tokens,  # Line2 for second sub-chunk
        ])

        # Max tokens per chunk allows header + one line, but not header + two lines
        max_tokens_per_chunk = header_tokens + line1_tokens + 5

        chunks = self.chat._split_patch(patch_content, max_tokens_per_chunk=max_tokens_per_chunk, model="gpt-test")

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith(file1_header.strip())) # strip because _split_patch strips file_diffs
        self.assertTrue(file1_line1.strip() in chunks[0])
        self.assertFalse(file1_line2.strip() in chunks[0])

        self.assertTrue(chunks[1].startswith(file1_header.strip()))
        self.assertTrue(file1_line2.strip() in chunks[1])
        self.assertFalse(file1_line1.strip() in chunks[1])

    def _calculate_mock_tokens(self, text, mock_encode_fn):
        return len(mock_encode_fn(text))


    @patch('chat.logger')
    @patch.dict(os.environ, {"MAX_TOKENS_OVERRIDE": "500"}) # Mock env var for model token limit
    def test_code_review_successful_chunking(self, mock_logger):
        # Setup: patch that needs chunking, mock OpenAI response for chunks
        long_patch = "diff --git a/file1.py b/file1.py\n" + "a" * 200 + "\ndiff --git a/file2.py b/file2.py\n" + "b" * 200
        # Mock _get_token_count to simulate token counts that trigger chunking
        # template_tokens + patch_tokens > max_total_tokens
        # Then _split_patch will be called. Let _split_patch use its own _get_token_count calls.

        # Mock parts of the token calculation in code_review
        self.chat._get_prompt_template_tokens = MagicMock(return_value=50) # Template tokens

        # Token counts for full patch, then for chunks by _split_patch
        # Full patch tokens: 200 (template + patch) > 150 (max_total_tokens for patch part)
        # Chunk1 tokens: 100
        # Chunk2 tokens: 100
        # self.chat._get_token_count needs to be versatile here.
        # For the initial check in code_review:
        #   - full_prompt_text_for_calc (template + markers + long_patch) -> e.g., 250 tokens
        # For _split_patch:
        #   - file1_diff (approx 100 + header) -> e.g., 110
        #   - file2_diff (approx 100 + header) -> e.g., 110

        # Simplified: Assume _split_patch works and returns 2 chunks.
        # We need to control the main token check in code_review.
        # Max total tokens for model: 500 (from env var)
        # Template tokens: 50
        # PROMPT_RESPONSE_BUFFER: 1024 (from chat.py) -> This buffer makes it hard to test chunking unless limit is high
        # Let's patch PROMPT_RESPONSE_BUFFER for this test
        with patch('chat.PROMPT_RESPONSE_BUFFER', 50):
            # max_patch_tokens_for_single_request = 500 - 50 - 50 = 400
            # If full_prompt_text_for_calc > 500, chunking happens.
            # Let's say full_prompt_text_for_calc is 510.

            # Mock _get_token_count specifically for the initial check and for _split_patch
            def mock_token_side_effect(text, model_name):
                if "--- BEGIN PATCH ---" in text and "END PATCH" in text and len(text) > 100: # Full prompt
                    return 510 # Triggers chunking
                elif "file1.py" in text: return 100 # Chunk 1 patch content
                elif "file2.py" in text: return 100 # Chunk 2 patch content
                else: return len(text) // 4 # Generic fallback for other calls (e.g. template itself)
            self.chat._get_token_count = MagicMock(side_effect=mock_token_side_effect)

            # Mock _split_patch to return predictable chunks
            chunk1_content = "diff --git a/file1.py b/file1.py\n" + "a" * 200
            chunk2_content = "diff --git a/file2.py b/file2.py\n" + "b" * 200
            self.chat._split_patch = MagicMock(return_value=[chunk1_content, chunk2_content])

            # Mock OpenAI API responses for each chunk
            response_chunk1 = {"lgtm": True, "review_comment": "Review for file1."}
            response_chunk2 = {"lgtm": False, "review_comment": "Review for file2 needs work."}

            mock_completion_chunk1 = MagicMock()
            mock_completion_chunk1.choices = [MagicMock(message=MagicMock(content=json.dumps(response_chunk1)))]
            mock_completion_chunk2 = MagicMock()
            mock_completion_chunk2.choices = [MagicMock(message=MagicMock(content=json.dumps(response_chunk2)))]

            self.chat.openai.chat.completions.create.side_effect = [mock_completion_chunk1, mock_completion_chunk2]

            result = self.chat.code_review(long_patch, model="gpt-test")

            self.assertEqual(self.chat.openai.chat.completions.create.call_count, 2)
            self.assertFalse(result["lgtm"])
            self.assertIn("--- Review for Chunk 1/2 ---", result["review_comment"])
            self.assertIn("Review for file1.", result["review_comment"])
            self.assertIn("--- Review for Chunk 2/2 ---", result["review_comment"])
            self.assertIn("Review for file2 needs work.", result["review_comment"])
            mock_logger.info.assert_any_call("Patch exceeds token limits (510 > 500). Splitting into chunks.")


    @patch('chat.logger')
    @patch.dict(os.environ, {"MAX_TOKENS_OVERRIDE": "500", "MAX_REVIEW_CHUNKS": "1"})
    def test_code_review_too_many_chunks(self, mock_logger):
        long_patch = "diff --git a/file1.py b/file1.py\n" + "a" * 200 + "\ndiff --git a/file2.py b/file2.py\n" + "b" * 200

        with patch('chat.PROMPT_RESPONSE_BUFFER', 50): # Keep buffer small for easier calculation
            # Simulate initial token calculation that triggers chunking
            self.chat._get_prompt_template_tokens = MagicMock(return_value=50)
            def mock_token_side_effect_initial(text, model_name):
                if "--- BEGIN PATCH ---" in text: return 510 # Force chunking
                return len(text) // 4
            self.chat._get_token_count = MagicMock(side_effect=mock_token_side_effect_initial)

            # Mock _split_patch to return more chunks than allowed
            self.chat._split_patch = MagicMock(return_value=["chunk1_content", "chunk2_content"]) # 2 chunks > MAX_REVIEW_CHUNKS (1)

            result = self.chat.code_review(long_patch, model="gpt-test")

            self.assertFalse(result["lgtm"])
            self.assertIn("Error: Diff is too large, resulting in 2 chunks (max 1).", result["review_comment"])
            self.chat.openai.chat.completions.create.assert_not_called() # API should not be called
            mock_logger.error.assert_any_call("Splitting resulted in 2 chunks, exceeding MAX_REVIEW_CHUNKS (1).")


if __name__ == '__main__':
    unittest.main()
