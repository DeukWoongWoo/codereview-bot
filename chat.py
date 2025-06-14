import json
import logging
import os
import time

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from typing import Any, Dict, Optional

# tiktoken import removed as chunking methods are being removed.
# yaml, os, time, json, logging, mlflow, MlflowException are used.
from openai import OpenAI, OpenAIError  # Ensure OpenAIError is imported

logger = logging.getLogger(__name__)

# Constants related to manual chunking were removed.


class Chat:
    """
    Manages interactions with the OpenAI Assistants API for code reviews.
    Handles assistant initialization, prompt loading, and code review generation.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initializes the Chat service and the OpenAI Assistant.

        Args:
            api_key: The OpenAI API key.
        """
        self.client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.assistant_id: Optional[str] = os.getenv("OPENAI_ASSISTANT_ID")
        self.assistant_name: str = os.getenv(
            "OPENAI_ASSISTANT_NAME", "GitLabCodeReviewAssistant"
        )
        # Model to be used when creating the assistant if not found by ID or name.
        self.model: str = os.getenv("MODEL", "gpt-4o")

        self.prompts: Dict[str, str] = self._load_prompts()
        self._initialize_assistant()

    def _initialize_assistant(self) -> None:
        """
        Initializes the OpenAI assistant.

        Attempts to retrieve an assistant by ID (if `OPENAI_ASSISTANT_ID` is set).
        If not found or ID not provided, it searches for an assistant by name
        (`OPENAI_ASSISTANT_NAME`). If still not found, it creates a new assistant.
        Updates `self.assistant_id` with the ID of the active assistant.

        Raises:
            Exception: If assistant initialization fails after all attempts.
        """
        logger.info("Initializing OpenAI Assistant...")
        if self.assistant_id:
            try:
                assistant = self.client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(
                    f"Successfully retrieved assistant with ID: {self.assistant_id}"
                )
                self.assistant_id = (
                    assistant.id
                )  # Ensure it's set from the retrieved object
                return
            except OpenAIError as e:
                logger.warning(
                    f"Failed to retrieve assistant with ID '{self.assistant_id}' due to API error: {type(e).__name__} - {e}. Falling back to finding by name or creating."
                )
                self.assistant_id = None  # Reset ID so we try to find or create

        if not self.assistant_id:  # Try to find by name
            logger.info(
                f"Attempting to find assistant by name: '{self.assistant_name}' (as ID was not provided or retrieval failed)."
            )
            try:
                assistants_response = self.client.beta.assistants.list(limit=100)
                found_by_name = False
                for assistant_data in assistants_response.data:
                    if assistant_data.name == self.assistant_name:
                        self.assistant_id = assistant_data.id
                        logger.info(
                            f"Found existing assistant by name: '{self.assistant_name}' with ID: {self.assistant_id}"
                        )
                        found_by_name = True
                        return
                if not found_by_name:
                    logger.info(
                        f"No assistant found with name '{self.assistant_name}'. Proceeding to create a new one."
                    )
            except OpenAIError as e:
                logger.error(
                    f"API error while listing assistants to find by name '{self.assistant_name}': {type(e).__name__} - {e}. Proceeding to create a new assistant."
                )

        if not self.assistant_id:  # If still no ID, create a new assistant
            logger.info(f"Creating a new assistant named '{self.assistant_name}'.")
            assistant_instructions: str = self.prompts.get(
                "default_review_prompt",
                "You are a helpful code review assistant. Please follow the provided JSON output format.",
            )
            try:
                new_assistant = self.client.beta.assistants.create(
                    name=self.assistant_name,
                    instructions=assistant_instructions,
                    model=self.model,
                    tools=[],  # No tools are defined for this assistant
                )
                self.assistant_id = new_assistant.id
                logger.info(
                    f"Successfully created new assistant '{self.assistant_name}' with ID: {self.assistant_id} and model: {self.model}"
                )
            except OpenAIError as e:
                logger.error(
                    f"Failed to create new assistant '{self.assistant_name}' due to API error: {type(e).__name__} - {e}"
                )
                # Specific error message for creation failure.
                raise Exception(
                    f"Fatal: OpenAI Assistant creation failed for '{self.assistant_name}'. Error: {type(e).__name__} - {e}"
                ) from e

        if not self.assistant_id:  # Should be unreachable if create failed and raised
            logger.critical(
                "Fatal: Assistant ID is None after all initialization attempts (retrieve, find, create)."
            )
            raise Exception("Fatal: Failed to initialize or create OpenAI assistant.")

    def _wait_for_run_completion(
        self, thread_id: str, run_id: str, timeout_seconds: int = 300
    ) -> Any:
        """
        Waits for an Assistant API run to complete by polling its status.

        Args:
            thread_id: The ID of the thread the run belongs to.
            run_id: The ID of the run to monitor.
            timeout_seconds: Maximum duration to wait for completion.

        Returns:
            The final Run object from the OpenAI API.

        Raises:
            TimeoutError: If the run does not reach a terminal state within
                          the specified timeout.
        """
        start_time = time.time()
        logger.info(
            f"Waiting for run {run_id} in thread {thread_id} to complete (timeout: {timeout_seconds}s)..."
        )
        while True:
            if time.time() - start_time > timeout_seconds:
                logger.error(
                    f"Run {run_id} in thread {thread_id} timed out after {timeout_seconds} seconds."
                )
                raise TimeoutError(
                    f"Run completion timed out for run_id: {run_id} in thread_id: {thread_id}"
                )

            try:
                run_obj = self.client.beta.threads.runs.retrieve( # Renamed variable
                    thread_id=thread_id, run_id=run_id
                )
            except OpenAIError as e:
                logger.error(
                    f"API error retrieving run {run_id} in thread {thread_id}: {type(e).__name__} - {e}. Retrying..."
                )
                time.sleep(5)  # Wait longer before retrying on API error
                continue

            logger.debug(f"Run {run_id} (Thread: {thread_id}) status: {run_obj.status}")
            if run_obj.status == "completed":
                logger.info(
                    f"Run {run_id} (Thread: {thread_id}) completed successfully."
                )
                return run_obj
            elif run_obj.status in ["failed", "cancelled", "expired"]:
                logger.error(
                    f"Run {run_id} (Thread: {thread_id}) ended with status: {run_obj.status}. Last error: {run_obj.last_error}"
                )
                return run_obj
            elif run_obj.status == "requires_action":
                logger.warning(
                    f"Run {run_id} (Thread: {thread_id}) requires action, but no tools are defined by this bot. This is an unexpected state."
                )
                return run_obj # Treat as a terminal state for this bot's purpose

            time.sleep(2)  # Poll interval for in-progress states (queued, in_progress)

    # Comment about old helper methods being removed is already present.

    def code_review(
        self,
        patch: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens_for_response: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Performs a code review on a given patch using the OpenAI Assistants API.

        Args:
            patch: The code patch (diff string) to be reviewed.
            model: (Optional) Specific model to use for this run, overriding assistant's default.
            temperature: (Optional) Sampling temperature for the run.
            top_p: (Optional) Nucleus sampling parameter for the run.
            max_tokens_for_response: (Optional) Max completion tokens for the run's response.

        Returns:
            A dictionary containing the review result:
            {
                "lgtm": bool,       // True if looks good, False otherwise or on error.
                "review_comment": str // Detailed review comment or error message.
            }
        """
        if not self.assistant_id:
            logger.error("Assistant not initialized. Cannot perform code review.")
            return {
                "lgtm": False,
                "review_comment": "Error: Assistant not initialized.",
            }
        if not patch:
            logger.info("Patch is empty. Skipping review.")
            return {"lgtm": True, "review_comment": ""}

        logger.info(f"Starting code review with Assistant ID: {self.assistant_id}")
        start_time = time.time()
        # Initialize thread_id and run_id for potential use in error logging
        thread_id_str: Optional[str] = None
        run_id_str: Optional[str] = None

        try:
            thread = self.client.beta.threads.create()
            thread_id_str = thread.id
            logger.info(f"Created new thread: {thread_id_str}")

            json_instruction: str = self.prompts.get(
                "json_format_requirement",
                'Output in JSON: {"lgtm": bool, "review_comment": str}',
            )
            message_content: str = (
                f"Please review this code patch:\n\n--- BEGIN PATCH ---\n{patch}\n--- END PATCH ---\n\n"
                f"Adhere strictly to this JSON output format:\n{json_instruction}"
            )

            self.client.beta.threads.messages.create(
                thread_id=thread_id_str, role="user", content=message_content
            )
            logger.info(f"Added message to thread {thread_id_str}")

            run_params: Dict[str, Any] = {"assistant_id": self.assistant_id}
            if temperature is not None:
                run_params["temperature"] = temperature
            if top_p is not None:
                run_params["top_p"] = top_p
            if max_tokens_for_response is not None:
                run_params["max_completion_tokens"] = max_tokens_for_response
            if model is not None:
                run_params["model"] = model  # Override assistant's model if provided

            run_obj = self.client.beta.threads.runs.create( # Renamed variable
                thread_id=thread_id_str, **run_params
            )
            run_id_str = run_obj.id
            logger.info(
                f"Created run {run_id_str} for thread {thread_id_str} with params: {run_params}"
            )

            run_status = self._wait_for_run_completion(thread_id_str, run_id_str)

            if run_status.status == "completed":
                messages_response = self.client.beta.threads.messages.list(
                    thread_id=thread_id_str, order="desc", limit=1
                )

                if (
                    messages_response.data
                    and messages_response.data[0].role == "assistant"
                ):
                    assistant_message = messages_response.data[0]
                    if (
                        assistant_message.content
                        and isinstance(assistant_message.content, list)
                        and assistant_message.content[0].type == "text"
                    ):
                        response_text: str = assistant_message.content[0].text.value
                        logger.info(f"Raw response from assistant: {response_text}")
                        try:
                            # Clean potential markdown fences
                            if response_text.startswith("```json\n"):
                                response_text = response_text[8:]
                            if response_text.endswith("\n```"):
                                response_text = response_text[:-4]
                            elif response_text.endswith("```"):
                                response_text = response_text[:-3]

                            review_data: Dict[str, Any] = json.loads(response_text)
                            return review_data
                        except json.JSONDecodeError as e:
                            snippet_length = 100
                            response_snippet = (
                                response_text[:snippet_length] + "..."
                                if len(response_text) > snippet_length
                                else response_text
                            )
                            logger.error(
                                f"Failed to parse JSON response from assistant for thread {thread_id_str}, run {run_id_str}: {e}. Response snippet: '{response_snippet}'"
                            )
                            return {
                                "lgtm": False,
                                "review_comment": f"Error: Could not parse AI response. Snippet: {response_snippet}",
                            }
                    else:
                        logger.error(
                            f"Assistant message content in thread {thread_id_str} is not text or is empty."
                        )
                        return {
                            "lgtm": False,
                            "review_comment": "Error: Assistant response was empty or not in the expected text format.",
                        }
                else:
                    logger.error(
                        f"No assistant messages found in thread {thread_id_str} after run {run_id_str} completion."
                    )
                    return {
                        "lgtm": False,
                        "review_comment": "Error: No response from assistant.",
                    }
            else:  # Run did not complete successfully
                error_message = f"Review failed for thread {thread_id_str}, run {run_id_str}. Run status: {run_status.status}."
                if run_status.last_error:
                    error_message += f" Last Error: Code: {run_status.last_error.code}, Message: {run_status.last_error.message}"
                logger.error(error_message)
                return {"lgtm": False, "review_comment": error_message}

        except OpenAIError as e:  # Catch specific OpenAI errors
            logger.error(
                f"OpenAI API error during code review (Thread: {thread_id_str}, Run: {run_id_str}): {type(e).__name__} - {e}",
                exc_info=True,
            )
            return {
                "lgtm": False,
                "review_comment": f"Error: OpenAI API error - {type(e).__name__}: {str(e)}",
            }
        except TimeoutError as e:  # Catch timeout from _wait_for_run_completion
            logger.error(
                f"Timeout during code review (Thread: {thread_id_str}, Run: {run_id_str}): {e}",
                exc_info=True,
            )
            return {
                "lgtm": False,
                "review_comment": f"Error: Code review timed out - {str(e)}",
            }
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                f"Unexpected error during code review (Thread: {thread_id_str}, Run: {run_id_str}): {type(e).__name__} - {e}",
                exc_info=True,
            )
            return {
                "lgtm": False,
                "review_comment": f"Error: An unexpected error occurred - {type(e).__name__}: {str(e)}",
            }
        finally:
            logger.info(
                f"Code review process for thread {thread_id_str} finished in {time.time() - start_time:.2f} seconds."
            )

    # Comment about old helper methods being removed is already present.

    def _load_prompts(self) -> Dict[str, str]:
        # Attempt to load from MLflow Prompt Registry first
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_prompt_name = os.getenv("MLFLOW_PROMPT_NAME")
        mlflow_prompt_version = os.getenv("MLFLOW_PROMPT_VERSION")  # Optional

        if mlflow_tracking_uri and mlflow_prompt_name:
            logger.info(
                f"Attempting to load prompts from MLflow Prompt Registry: Name='{mlflow_prompt_name}', Version='{mlflow_prompt_version or 'latest'}'"
            )
            try:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

                prompt_uri = f"prompts:/{mlflow_prompt_name}"
                if mlflow_prompt_version:
                    prompt_uri += f"/{mlflow_prompt_version}"

                # Use mlflow.prompts.load_prompt as per the updated requirement
                prompt_yaml_str = mlflow.prompts.load_prompt(uri=prompt_uri)

                loaded_prompts = yaml.safe_load(prompt_yaml_str)

                # Validate the loaded prompts
                if (
                    isinstance(loaded_prompts, dict)
                    and "default_review_prompt" in loaded_prompts
                    and "json_format_requirement" in loaded_prompts
                ):
                    logger.info(
                        "Successfully loaded and validated prompts from MLflow Prompt Registry."
                    )
                    return loaded_prompts
                else:
                    logger.warning(
                        "Prompts loaded from MLflow Prompt Registry are invalid or missing expected keys. Proceeding to fallbacks."
                    )

            except MlflowException as e:
                logger.error(
                    f"MLflowException while loading from Prompt Registry (Name: {mlflow_prompt_name}): {type(e).__name__} - {e}"
                )
            except yaml.YAMLError as e:
                logger.error(
                    f"YAML parsing error for prompts from MLflow Prompt Registry (Name: {mlflow_prompt_name}): {e}"
                )
            except (
                AttributeError
            ) as e:  # Catch if mlflow.prompts or load_prompt is not available
                logger.error(
                    f"AttributeError with MLflow Prompt Registry (Name: {mlflow_prompt_name}, check MLflow version/API): {type(e).__name__} - {e}"
                )
            except (
                Exception
            ) as e:  # Catch any other unexpected error during MLflow loading
                logger.error(
                    f"Unexpected error loading prompts from MLflow Prompt Registry (Name: {mlflow_prompt_name}): {type(e).__name__} - {e}"
                )
            logger.info(
                f"Falling back from MLflow Prompt Registry for prompt: {mlflow_prompt_name}"
            )

        # Fallback to local prompts.yaml
        try:
            logger.info("Attempting to load prompts from local prompts.yaml file.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, "prompts.yaml")
            if os.path.exists(
                prompts_path
            ):  # Check if file exists before trying to open
                with open(prompts_path, "r") as f:
                    prompts = yaml.safe_load(f)
                logger.info("Successfully loaded prompts from local prompts.yaml.")
                return prompts
            else:
                logger.info("Local prompts.yaml not found.")
        except Exception as e:
            logger.error(f"Error loading local prompts.yaml: {e}")

        # Fallback to hardcoded defaults
        logger.warning("Falling back to hardcoded default prompts.")
        return {
            "default_review_prompt": "Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.",
            "json_format_requirement": 'Provide your feedback in a strict JSON format with the following structure:\n{\n    "lgtm": boolean, // true if the code looks good to merge, false if there are concerns\n    "review_comment": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object.',
        }

    # The _load_prompts method is now defined above code_review
    # This section will be replaced by the new code_review method.
    # The old helper methods (_get_token_count, _generate_prompt, etc.) will be removed in the next step.
