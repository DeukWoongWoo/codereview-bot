import json
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import yaml

# It's good practice to import the module under test after setting up potential mocks if needed,
# or ensure mocks are applied correctly. Here, we're importing Chat from the 'chat' module.
# Assuming chat.py is in the root directory, and tests are in tests/
# We might need to adjust sys.path if running tests directly and chat.py isn't found.
# However, typical test runners (like pytest or unittest discovery) handle this.
try:
    from chat import \
        Chat  # Assuming chat.py is in the parent directory or PYTHONPATH
except ImportError:
    # Adjust path for local testing if necessary
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from chat import \
        ACTUAL_DEFAULT_PROMPTS as \
        CHAT_ACTUAL_DEFAULT_PROMPTS  # Import for comparison
    from chat import Chat

    # Make pandas an optional import for tests if chat.py treats it as such
    try:
        import pandas as pd
    except ImportError:
        pd = None  # Mock or skip pandas-dependent tests if not available
except ImportError:
    # Adjust path for local testing if necessary
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from chat import ACTUAL_DEFAULT_PROMPTS as CHAT_ACTUAL_DEFAULT_PROMPTS
    from chat import Chat

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
ACTUAL_DEFAULT_PROMPTS_IN_TEST_FILE_FOR_REFERENCE_ONLY = {  # Renamed to avoid confusion
    "default_review_prompt": "Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.",
    "json_format_requirement": 'Provide your feedback in a strict JSON format with the following structure:\n{\n    "lgtm": boolean, // true if the code looks good to merge, false if there are concerns\n    "review_comment": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object.',
}

SAMPLE_YAML_PROMPTS = {
    "default_review_prompt": "Custom review prompt from YAML.",
    "json_format_requirement": "Custom JSON format from YAML.",
}
SAMPLE_YAML_CONTENT = yaml.dump(SAMPLE_YAML_PROMPTS)

# Prompts for MLflow testing
MLFLOW_PROMPTS = {
    "default_review_prompt": "MLflow artifact prompt.",
    "json_format_requirement": "MLflow JSON format.",
}
MLFLOW_PROMPTS_YAML_CONTENT = yaml.dump(MLFLOW_PROMPTS)

# TestChatBase removed as its tests (_generate_prompt, old code_review) are for obsolete methods


# Separate class for testing the _load_prompts logic with MLflow and fallbacks
class TestChatPromptLoading(unittest.TestCase):

    def setUp(self):
        # These mocks will be started and stopped for each test method via decorators or context managers
        self.mock_mlflow_set_uri = patch("chat.mlflow.set_tracking_uri").start()
        # Removed: self.mock_mlflow_download = patch('chat.mlflow.artifacts.download_artifacts').start()
        # Removed: self.mock_mlflow_load_model = patch('chat.mlflow.pyfunc.load_model').start()
        self.mock_mlflow_load_prompt = patch(
            "chat.mlflow.prompts.load_prompt"
        ).start()  # Updated mock target

        self.mock_os_path_exists = patch("os.path.exists").start()
        self.mock_open_builtin = patch("builtins.open", new_callable=mock_open).start()

        # Mock logger to check log messages
        self.mock_logger = patch("chat.logger").start()

        # Ensure pandas is available for model loading tests, or skip them
        if pd is None:
            self.skip_pandas_tests = True  # Flag to skip model loading tests
        else:
            self.skip_pandas_tests = False

    def tearDown(self):
        patch.stopall()  # Stops all patches started with patch().start()

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_NAME": "test_prompt",
            # No version - tests default fetching
        },
    )
    def test_load_prompts_mlflow_registry_success_no_version(self):
        self.mock_mlflow_load_prompt.return_value = MLFLOW_PROMPTS_YAML_CONTENT

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, MLFLOW_PROMPTS)
        self.mock_mlflow_set_uri.assert_called_once_with("http://fake-mlflow-uri")
        self.mock_mlflow_load_prompt.assert_called_once_with(uri="prompts:/test_prompt")
        self.mock_logger.info.assert_any_call(
            "Successfully loaded and validated prompts from MLflow Prompt Registry."
        )
        self.mock_os_path_exists.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_NAME": "test_prompt_versioned",
            "MLFLOW_PROMPT_VERSION": "2",
        },
    )
    def test_load_prompts_mlflow_registry_success_with_version(self):
        self.mock_mlflow_load_prompt.return_value = MLFLOW_PROMPTS_YAML_CONTENT

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, MLFLOW_PROMPTS)
        self.mock_mlflow_set_uri.assert_called_once_with("http://fake-mlflow-uri")
        self.mock_mlflow_load_prompt.assert_called_once_with(
            uri="prompts:/test_prompt_versioned/2"
        )
        self.mock_logger.info.assert_any_call(
            "Successfully loaded and validated prompts from MLflow Prompt Registry."
        )

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_NAME": "test_prompt_fail",
        },
    )
    def test_load_prompts_mlflow_registry_failure_fallback_to_yaml(self):
        self.mock_mlflow_load_prompt.side_effect = MlflowException("Registry error")
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(
            read_data=SAMPLE_YAML_CONTENT
        ).return_value

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        # Verify more specific logging
        self.mock_logger.error.assert_any_call(
            "MLflowException while loading from Prompt Registry (Name: test_prompt_fail): MlflowException - Registry error"
        )
        self.mock_logger.info.assert_any_call(
            "Falling back from MLflow Prompt Registry for prompt: test_prompt_fail"
        )
        self.mock_logger.info.assert_any_call(
            "Successfully loaded prompts from local prompts.yaml."
        )

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_NAME": "test_prompt_invalid_yaml",
        },
    )
    def test_load_prompts_mlflow_registry_invalid_yaml_fallback_to_yaml(self):
        self.mock_mlflow_load_prompt.return_value = "this: is: invalid: yaml"
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(
            read_data=SAMPLE_YAML_CONTENT
        ).return_value

        chat_instance = Chat(api_key="key")
        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        # Check for specific log message including the prompt name
        # The actual error message from yaml.YAMLError can be complex, so we match the start.
        self.assertTrue(
            any(
                f"YAML parsing error for prompts from MLflow Prompt Registry (Name: test_prompt_invalid_yaml):" in call_args[0][0]
                for call_args in self.mock_logger.error.call_args_list
            )
        )
        self.mock_logger.info.assert_any_call(
            "Falling back from MLflow Prompt Registry for prompt: test_prompt_invalid_yaml"
        )
        self.mock_logger.info.assert_any_call(
            "Successfully loaded prompts from local prompts.yaml."
        )

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_NAME": "test_prompt_invalid_structure",
        },
    )
    def test_load_prompts_mlflow_registry_invalid_structure_fallback_to_yaml(self):
        invalid_structure_prompts = {"only_one_key": "value"}
        self.mock_mlflow_load_prompt.return_value = yaml.dump(invalid_structure_prompts)
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(
            read_data=SAMPLE_YAML_CONTENT
        ).return_value

        chat_instance = Chat(api_key="key")
        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_logger.warning.assert_any_call(
            "Prompts loaded from MLflow Prompt Registry are invalid or missing expected keys. Proceeding to fallbacks."
        )
        self.mock_logger.info.assert_any_call(
            "Falling back from MLflow Prompt Registry for prompt: test_prompt_invalid_structure"
        )
        self.mock_logger.info.assert_any_call(
            "Successfully loaded prompts from local prompts.yaml."
        )

    @patch.dict(
        os.environ, {"MLFLOW_TRACKING_URI": "http://fake-mlflow-uri"}
    )  # MLFLOW_PROMPT_NAME is missing
    def test_load_prompts_mlflow_not_fully_configured_fallback_to_yaml(self):
        self.mock_os_path_exists.return_value = True
        self.mock_open_builtin.return_value = mock_open(
            read_data=SAMPLE_YAML_CONTENT
        ).return_value

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
        self.mock_mlflow_set_uri.assert_not_called()
        self.mock_logger.info.assert_any_call(
            "Successfully loaded prompts from local prompts.yaml."
        )

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",
            "MLFLOW_PROMPT_SOURCE": "artifact",
            # Missing RUN_ID and ARTIFACT_PATH intentionally
        },
    )
    # This test was for the old artifact loading logic and is no longer relevant
    # def test_load_prompts_mlflow_artifact_misconfigured_fallback_to_yaml(self):
    #     self.mock_os_path_exists.return_value = True # prompts.yaml exists
    #     self.mock_open_builtin.return_value = mock_open(read_data=SAMPLE_YAML_CONTENT).return_value

    #     chat_instance = Chat(api_key="key")
    #     self.assertEqual(chat_instance.prompts, SAMPLE_YAML_PROMPTS)
    #     self.mock_logger.warning.assert_any_call("MLFLOW_RUN_ID and MLFLOW_ARTIFACT_PATH must be set for source 'artifact'.")
    #     self.mock_logger.info.assert_any_call("Successfully loaded prompts from local prompts.yaml.")

    @patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-mlflow-uri",  # Configured for MLflow
            "MLFLOW_PROMPT_NAME": "test_prompt_hardcoded_fallback",
        },
    )
    def test_load_prompts_all_fallbacks_to_hardcoded(self):
        self.mock_mlflow_load_prompt.side_effect = MlflowException(
            "Registry error"
        )  # Corrected to use the new mock
        self.mock_os_path_exists.return_value = False  # prompts.yaml does not exist
        # mock_open will not be called for local prompts.yaml if os.path.exists is false

        chat_instance = Chat(api_key="key")

        self.assertEqual(chat_instance.prompts, CHAT_ACTUAL_DEFAULT_PROMPTS)
        self.mock_logger.error.assert_any_call(
            "MLflowException while loading from Prompt Registry (Name: test_prompt_hardcoded_fallback): MlflowException - Registry error"
        )
        self.mock_logger.info.assert_any_call(
            "Falling back from MLflow Prompt Registry for prompt: test_prompt_hardcoded_fallback"
        )
        self.mock_logger.info.assert_any_call("Local prompts.yaml not found.")
        self.mock_logger.warning.assert_any_call(
            "Falling back to hardcoded default prompts."
        )


# Removing TestChatCodeReviewChunking as its tests (_get_token_count, _split_patch, old code_review chunking) are for obsolete methods

# TestChatPromptLoading remains relevant and is kept.


# --- New Test Class for Assistants API ---
TEST_PROMPTS_FOR_ASSISTANT = {
    "default_review_prompt": "Test assistant instructions for review.",
    "json_format_requirement": 'Output in JSON: {"lgtm": bool, "review_comment": "your_review_here"}',
}


class TestChatAssistantsAPI(unittest.TestCase):
    def setUp(self):
        # Patch OpenAI client initialized in Chat
        self.mock_openai_client_patch = patch("chat.OpenAI")
        self.MockOpenAIClass = self.mock_openai_client_patch.start()
        self.mock_client_instance = self.MockOpenAIClass.return_value
        self.mock_client_instance.beta = MagicMock()  # Mock the .beta attribute

        # Mock _load_prompts for Chat instance
        self.mock_load_prompts_patch = patch("chat.Chat._load_prompts")
        self.mock_load_prompts = self.mock_load_prompts_patch.start()
        self.mock_load_prompts.return_value = TEST_PROMPTS_FOR_ASSISTANT

        # Mock time.sleep to speed up tests
        self.mock_time_sleep = patch("time.sleep").start()

        # Mock logger to check log messages
        self.mock_logger = patch("chat.logger").start()

        # Default assistant name for tests
        self.assistant_name = "GitLabCodeReviewAssistant"  # Matches default in chat.py

    def tearDown(self):
        patch.stopall()

    # --- Tests for _initialize_assistant ---
    @patch.dict(os.environ, {"OPENAI_ASSISTANT_ID": "asst_env_id_123"})
    def test_initialize_assistant_from_env_id_success(self):
        mock_retrieved_assistant = MagicMock(id="asst_env_id_123", name="EnvAssistant")
        self.mock_client_instance.beta.assistants.retrieve.return_value = (
            mock_retrieved_assistant
        )

        # Pass a dummy model, actual model comes from env or chat.py default in Chat.__init__
        chat_instance = Chat(api_key="test_key")

        self.assertEqual(chat_instance.assistant_id, "asst_env_id_123")
        self.mock_client_instance.beta.assistants.retrieve.assert_called_once_with(
            assistant_id="asst_env_id_123"
        )
        self.mock_client_instance.beta.assistants.list.assert_not_called()
        self.mock_client_instance.beta.assistants.create.assert_not_called()
        # Verify initial log and success log
        self.mock_logger.info.assert_any_call("Initializing OpenAI Assistant...")
        self.mock_logger.info.assert_any_call(
            "Successfully retrieved assistant with ID: asst_env_id_123"
        )

    @patch.dict(
        os.environ,
        {
            "OPENAI_ASSISTANT_ID": "asst_env_id_fail",
            "OPENAI_ASSISTANT_NAME": "TestAssistantCreate",
            "MODEL": "gpt-test-model",
        },
    )
    def test_initialize_assistant_from_env_id_failure_then_create(self):
        self.mock_client_instance.beta.assistants.retrieve.side_effect = OpenAIError(
            "Retrieval failed"
        )

        mock_empty_list_response = MagicMock()
        mock_empty_list_response.data = []
        self.mock_client_instance.beta.assistants.list.return_value = (
            mock_empty_list_response
        )

        mock_created_assistant = MagicMock(
            id="asst_created_new_456", name="TestAssistantCreate"
        )
        self.mock_client_instance.beta.assistants.create.return_value = (
            mock_created_assistant
        )

        chat_instance = Chat(api_key="test_key")  # Uses MODEL from patched env

        self.assertEqual(chat_instance.assistant_id, "asst_created_new_456")
        self.mock_client_instance.beta.assistants.retrieve.assert_called_once_with(
            assistant_id="asst_env_id_fail"
        )
        self.mock_client_instance.beta.assistants.list.assert_called_once()
        self.mock_client_instance.beta.assistants.create.assert_called_once_with(
            name="TestAssistantCreate",  # From env
            instructions=TEST_PROMPTS_FOR_ASSISTANT["default_review_prompt"],
            model="gpt-test-model",  # From env
            instructions=TEST_PROMPTS_FOR_ASSISTANT["default_review_prompt"],
            model="gpt-test-model",  # From env
            tools=[],
        )
        # Check specific log messages for each step of the fallback
        self.mock_logger.warning.assert_any_call(
            "Failed to retrieve assistant with ID 'asst_env_id_fail' due to API error: OpenAIError - Retrieval failed. Falling back to finding by name or creating."
        )
        self.mock_logger.info.assert_any_call(
            f"Attempting to find assistant by name: 'TestAssistantCreate' (as ID was not provided or retrieval failed)."
        )
        self.mock_logger.info.assert_any_call(
            f"No assistant found with name 'TestAssistantCreate'. Proceeding to create a new one."
        )
        self.mock_logger.info.assert_any_call(
            "Creating a new assistant named 'TestAssistantCreate'."
        )
        self.mock_logger.info.assert_any_call(
            "Successfully created new assistant 'TestAssistantCreate' with ID: asst_created_new_456 and model: gpt-test-model"
        )

    @patch.dict(
        os.environ,
        {"OPENAI_ASSISTANT_ID": "", "OPENAI_ASSISTANT_NAME": "ExistingAssistant"},
    )
    def test_initialize_assistant_find_by_name_success(self):
        mock_assistant_in_list = MagicMock(
            id="asst_found_by_name_789", name="ExistingAssistant"
        )
        mock_list_response = MagicMock()
        mock_list_response.data = [
            MagicMock(id="other_asst_000", name="Other"),
            mock_assistant_in_list,
        ]
        self.mock_client_instance.beta.assistants.list.return_value = mock_list_response

        chat_instance = Chat(api_key="test_key")

        self.assertEqual(chat_instance.assistant_id, "asst_found_by_name_789")
        self.mock_client_instance.beta.assistants.retrieve.assert_not_called()
        self.mock_client_instance.beta.assistants.list.assert_called_once()
        self.mock_client_instance.beta.assistants.create.assert_not_called()
        self.mock_logger.info.assert_any_call(
            f"Attempting to find assistant by name: 'ExistingAssistant' (as ID was not provided or retrieval failed)."
        )
        self.mock_logger.info.assert_any_call(
            "Found existing assistant by name: 'ExistingAssistant' with ID: asst_found_by_name_789"
        )

    @patch.dict(
        os.environ,
        {"OPENAI_ASSISTANT_ID": "", "OPENAI_ASSISTANT_NAME": "NewUniqueAssistant"},
    )
    def test_initialize_assistant_create_new_if_not_found(self):
        mock_empty_list_response = MagicMock()
        mock_empty_list_response.data = []
        self.mock_client_instance.beta.assistants.list.return_value = (
            mock_empty_list_response
        )

        mock_created_assistant = MagicMock(
            id="asst_newly_created_111", name="NewUniqueAssistant"
        )
        self.mock_client_instance.beta.assistants.create.return_value = (
            mock_created_assistant
        )

        # Ensure MODEL env var is not set to test chat.py's default model for assistant creation
        with patch.dict(
            os.environ, {"MODEL": ""}, clear=True
        ):  # Temporarily clear MODEL for this test scope
            # Need to set OPENAI_ASSISTANT_NAME again as clear=True wipes it
            os.environ["OPENAI_ASSISTANT_NAME"] = "NewUniqueAssistant"
            chat_instance = Chat(api_key="test_key")

        self.assertEqual(chat_instance.assistant_id, "asst_newly_created_111")
        self.mock_client_instance.beta.assistants.create.assert_called_once_with(
            name="NewUniqueAssistant",
            instructions=TEST_PROMPTS_FOR_ASSISTANT["default_review_prompt"],
            model="gpt-4o",  # Default from Chat class
            tools=[],
        )
        self.assertEqual(chat_instance.assistant_name, "NewUniqueAssistant")
        self.mock_logger.info.assert_any_call(
            f"Attempting to find assistant by name: 'NewUniqueAssistant' (as ID was not provided or retrieval failed)."
        )
        self.mock_logger.info.assert_any_call(
            f"No assistant found with name 'NewUniqueAssistant'. Proceeding to create a new one."
        )
        self.mock_logger.info.assert_any_call(
            "Creating a new assistant named 'NewUniqueAssistant'."
        )
        self.mock_logger.info.assert_any_call(
            f"Successfully created new assistant 'NewUniqueAssistant' with ID: {mock_created_assistant.id} and model: gpt-4o"
        )

    @patch.dict(
        os.environ,
        {"OPENAI_ASSISTANT_ID": "", "OPENAI_ASSISTANT_NAME": "FailAssistant"},
    )
    def test_initialize_assistant_all_fail_raises_exception(self):
        self.mock_client_instance.beta.assistants.retrieve.side_effect = OpenAIError(
            "Retrieve failed"
        )
        self.mock_client_instance.beta.assistants.list.side_effect = OpenAIError(
            "List failed"
        )
        self.mock_client_instance.beta.assistants.create.side_effect = OpenAIError(
            "Create failed"
        )

        with self.assertRaisesRegex(
            Exception,
            # Matching the more specific error message from the refactored code
            "Fatal: OpenAI Assistant creation failed for 'FailAssistant'. Error: OpenAIError - Create failed",
        ):
            Chat(api_key="test_key")

        # Check for the sequence of logs indicating attempts and failures
        self.mock_logger.info.assert_any_call("Initializing OpenAI Assistant...")
        # It tries to find by name first when ID is empty
        self.mock_logger.info.assert_any_call("Attempting to find assistant by name: 'FailAssistant' (as ID was not provided or retrieval failed).")
        self.mock_logger.error.assert_any_call("API error while listing assistants to find by name 'FailAssistant': OpenAIError - List failed. Proceeding to create a new assistant.")
        self.mock_logger.info.assert_any_call("Creating a new assistant named 'FailAssistant'.")
        self.mock_logger.error.assert_any_call("Failed to create new assistant 'FailAssistant' due to API error: OpenAIError - Create failed")

    # --- Tests for code_review and _wait_for_run_completion ---
    def test_code_review_success(self):
        # Chat instance is created, _initialize_assistant is called. Let's assume it sets an ID.
        # For this test, we can directly set chat_instance.assistant_id after Chat() if needed,
        # or ensure setUp's mocks for _initialize_assistant lead to a valid ID.
        # Here, _load_prompts is mocked, so assistant creation inside _initialize_assistant uses default instructions.

        # To ensure assistant_id is set before code_review is called:
        # One way: mock _initialize_assistant itself
        with patch.object(
            Chat, "_initialize_assistant", autospec=True
        ) as mock_init_asst:
            chat_instance = Chat(api_key="test_key")
            chat_instance.assistant_id = (
                "asst_test_id"  # Manually set for clarity in test focus
            )
            mock_init_asst.assert_called_once()  # Ensure it was called by __init__

        mock_thread = MagicMock(id="thread_abc")
        self.mock_client_instance.beta.threads.create.return_value = mock_thread

        mock_run_params = {
            "assistant_id": "asst_test_id",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_completion_tokens": 500,
            "model": "gpt-override-model",  # Test model override
        }
        mock_run_create_obj = MagicMock(id="run_xyz", status="queued", **run_params)
        self.mock_client_instance.beta.threads.runs.create.return_value = (
            mock_run_create_obj
        )

        mock_run_in_progress = MagicMock(id="run_xyz", status="in_progress")
        mock_run_completed = MagicMock(id="run_xyz", status="completed")
        self.mock_client_instance.beta.threads.runs.retrieve.side_effect = [
            mock_run_in_progress,
            mock_run_completed,
        ]

        mock_assistant_message_content = MagicMock(
            type="text",
            text=MagicMock(
                value=json.dumps({"lgtm": True, "review_comment": "All good!"})
            ),
        )
        mock_assistant_message = MagicMock(
            role="assistant", content=[mock_assistant_message_content]
        )
        mock_messages_list = MagicMock(data=[mock_assistant_message])
        self.mock_client_instance.beta.threads.messages.list.return_value = (
            mock_messages_list
        )

        review = chat_instance.code_review(
            "test patch",
            model="gpt-override-model",
            temperature=0.1,
            top_p=0.9,
            max_tokens_for_response=500,
        )

        self.assertEqual(review, {"lgtm": True, "review_comment": "All good!"})
        self.mock_client_instance.beta.threads.create.assert_called_once()
        self.mock_client_instance.beta.threads.messages.create.assert_called_once()
        self.mock_client_instance.beta.threads.runs.create.assert_called_once_with(
            thread_id=mock_thread.id, **mock_run_params
        )
        self.assertEqual(
            self.mock_client_instance.beta.threads.runs.retrieve.call_count, 2
        )
        self.mock_client_instance.beta.threads.messages.list.assert_called_once_with(
            thread_id=mock_thread.id, order="desc", limit=1
        )
        # Logging assertions for successful review
        self.mock_logger.info.assert_any_call(f"Starting code review with Assistant ID: {chat_instance.assistant_id}")
        self.mock_logger.info.assert_any_call(f"Created new thread: {mock_thread.id}")
        self.mock_logger.info.assert_any_call(f"Added message to thread {mock_thread.id}")
        self.mock_logger.info.assert_any_call(f"Created run {mock_run_create_obj.id} for thread {mock_thread.id} with params: {mock_run_params}")
        self.mock_logger.info.assert_any_call(f"Waiting for run {mock_run_create_obj.id} in thread {mock_thread.id} to complete (timeout: 300s)...")
        self.mock_logger.info.assert_any_call(f"Run {mock_run_completed.id} (Thread: {mock_thread.id}) completed successfully.")
        self.mock_logger.info.assert_any_call(f"Raw response from assistant: {json.dumps({'lgtm': True, 'review_comment': 'All good!'})}")
        self.mock_logger.info.assert_any_call(f"Code review process for thread {mock_thread.id} finished in {unittest.mock.ANY} seconds.")


    def test_code_review_run_fails(self):
        with patch.object(
            Chat, "_initialize_assistant", autospec=True
        ):  # Mock init to simplify
            chat_instance = Chat(api_key="test_key")
            chat_instance.assistant_id = "asst_test_id"

        self.mock_client_instance.beta.threads.create.return_value = MagicMock(
            id="thread_fail"
        )
        self.mock_client_instance.beta.threads.runs.create.return_value = MagicMock(
            id="run_fail", status="queued"
        )

        mock_run_failed_error = MagicMock(
            code="server_error", message="Server exploded"
        )
        mock_run_failed = MagicMock(
            id="run_fail", status="failed", last_error=mock_run_failed_error
        )
        self.mock_client_instance.beta.threads.runs.retrieve.return_value = (
            mock_run_failed
        )

        review = chat_instance.code_review("test patch for failure")

        self.assertFalse(review["lgtm"])
        self.assertIn("Review failed. Run status: failed.", review["review_comment"])
        self.assertIn("Server exploded", review["review_comment"])
        # Use IDs from mock_run_failed and its (assumed) thread_id attribute or the one from create call
        # For simplicity, let's assume self.mock_client_instance.beta.threads.create.return_value.id was "thread_fail"
        # and self.mock_client_instance.beta.threads.runs.create.return_value.id was "run_fail"
        self.mock_logger.error.assert_any_call(
            f"Run {self.mock_client_instance.beta.threads.runs.create.return_value.id} (Thread: {self.mock_client_instance.beta.threads.create.return_value.id}) ended with status: failed. Last error: {mock_run_failed_error}"
        )
        self.mock_logger.info.assert_any_call(f"Code review process for thread {self.mock_client_instance.beta.threads.create.return_value.id} finished in {unittest.mock.ANY} seconds.")

    @patch("time.time")
    def test_code_review_run_timeout_in_wait_method(self, mock_time_time_func):
        # Test _wait_for_run_completion directly for timeout
        with patch.object(Chat, "_initialize_assistant", autospec=True):
            chat_instance = Chat(api_key="test_key")
            # No need to set assistant_id as we are testing _wait_for_run_completion directly

        mock_start_time = 1000.0
        mock_time_time_func.side_effect = [
            mock_start_time,  # Call in _wait_for_run_completion start
            mock_start_time + 1,  # First loop check
            mock_start_time + 6,  # Second loop check, timeout (5s + 1s margin)
        ]
        self.mock_client_instance.beta.threads.runs.retrieve.return_value = MagicMock(
            status="in_progress"
        )

        with self.assertRaises(TimeoutError):
            chat_instance._wait_for_run_completion(mock_thread_id, mock_run_id, timeout_seconds=5)
        self.mock_logger.error.assert_any_call(f"Run {mock_run_id} in thread {mock_thread_id} timed out after 5 seconds.")

    def test_code_review_catches_timeout_from_wait(self):
        # Test that code_review catches TimeoutError from _wait_for_run_completion
        mock_thread_id_val = "thread_timeout_catch"
        mock_run_id_val = "run_timeout_catch"
        with patch.object(Chat, '_initialize_assistant', autospec=True), \
             patch.object(Chat, '_wait_for_run_completion', side_effect=TimeoutError("Simulated timeout in wait")) as mock_wait:
            chat_instance = Chat(api_key="test_key")
            chat_instance.assistant_id = "asst_test_id"

            self.mock_client_instance.beta.threads.create.return_value = MagicMock(id=mock_thread_id_val)
            self.mock_client_instance.beta.threads.runs.create.return_value = MagicMock(id=mock_run_id_val, status="queued")

            review = chat_instance.code_review("test patch for caught timeout")

            self.assertFalse(review["lgtm"])
            self.assertIn(
                "Error: Code review timed out - Simulated timeout in wait",
                review["review_comment"],
            )
            mock_wait.assert_called_once()
            self.mock_logger.error.assert_any_call(f"Timeout during code review (Thread: {mock_thread_id_val}, Run: {mock_run_id_val}): Simulated timeout in wait", exc_info=True)
            self.mock_logger.info.assert_any_call(f"Code review process for thread {mock_thread_id_val} finished in {unittest.mock.ANY} seconds.")


    def test_code_review_assistant_returns_invalid_json(self):
        with patch.object(Chat, "_initialize_assistant", autospec=True):
            chat_instance = Chat(api_key="test_key")
            chat_instance.assistant_id = "asst_test_id"

        self.mock_client_instance.beta.threads.create.return_value = MagicMock(
            id="thread_json_err"
        )
        mock_run_completed = MagicMock(id="run_json_err", status="completed")
        self.mock_client_instance.beta.threads.runs.create.return_value = (
            mock_run_completed
        )
        self.mock_client_instance.beta.threads.runs.retrieve.return_value = (
            mock_run_completed
        )

        invalid_json_string = "This is not JSON { definitely not"
        mock_assistant_message_content = MagicMock(
            type="text", text=MagicMock(value=invalid_json_string)
        )
        mock_assistant_message = MagicMock(
            role="assistant", content=[mock_assistant_message_content]
        )
        mock_messages_list = MagicMock(data=[mock_assistant_message])
        self.mock_client_instance.beta.threads.messages.list.return_value = (
            mock_messages_list
        )

        review = chat_instance.code_review("test patch for invalid json")

        self.assertFalse(review["lgtm"])
        self.assertIn("Error: Could not parse AI response.", review["review_comment"])
        self.assertIn(f"Raw: {invalid_json_string}", review["review_comment"])
        self.mock_logger.error.assert_any_call(
            f"Failed to parse JSON response from assistant for thread thread_json_err, run run_json_err: Expecting value: line 1 column 1 (char 0). Response snippet: '{invalid_json_string[:100]}...'"
        )
        self.mock_logger.info.assert_any_call(f"Code review process for thread thread_json_err finished in {unittest.mock.ANY} seconds.")


    def test_code_review_empty_patch(self):
        with patch.object(Chat, "_initialize_assistant", autospec=True): # Keep this simple as it's not the focus
            chat_instance = Chat(api_key="test_key")
            chat_instance.assistant_id = "asst_test_id" # Manually set for test focus

        review = chat_instance.code_review("")

        self.assertTrue(review["lgtm"])
        self.assertEqual(review["review_comment"], "")
        self.mock_client_instance.beta.threads.create.assert_not_called()
        self.mock_logger.info.assert_any_call("Patch is empty. Skipping review.")


if __name__ == "__main__":
    unittest.main()
