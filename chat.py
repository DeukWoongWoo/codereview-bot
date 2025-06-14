from openai import OpenAI
import os
import time
import yaml
import json
import logging
import mlflow
from mlflow.exceptions import MlflowException
# tiktoken import removed as chunking methods are being removed.
# yaml, os, time, json, logging, mlflow, MlflowException are used.
from openai import OpenAI, OpenAIError # Ensure OpenAIError is imported

logger = logging.getLogger(__name__)

# Removed: MODEL_TOKEN_LIMITS, DEFAULT_ENCODER, DEFAULT_MAX_REVIEW_CHUNKS, PROMPT_RESPONSE_BUFFER
# These constants were related to manual chunking.

class Chat:
    def __init__(self, api_key: str):
        self.client = OpenAI( # Changed self.openai to self.client
            api_key=api_key,
            base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        )
        self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        self.assistant_name = os.getenv("OPENAI_ASSISTANT_NAME", "GitLabCodeReviewAssistant")
        self.model = os.getenv("MODEL", "gpt-4o") # Default to a newer model

        self.prompts = self._load_prompts() # For assistant instructions & JSON format requirement
        self._initialize_assistant()

    def _initialize_assistant(self):
        logger.info("Initializing OpenAI Assistant...")
        if self.assistant_id:
            try:
                assistant = self.client.beta.assistants.retrieve(assistant_id=self.assistant_id)
                logger.info(f"Successfully retrieved assistant with ID: {self.assistant_id}")
                self.assistant_id = assistant.id # Ensure it's set from the retrieved object
                return
            except OpenAIError as e:
                logger.warning(f"Failed to retrieve assistant with ID '{self.assistant_id}': {e}. Will try to find by name or create a new one.")
                self.assistant_id = None # Reset ID so we try to find or create

        if not self.assistant_id: # Try to find by name
            logger.info(f"No Assistant ID from environment, trying to find assistant by name: '{self.assistant_name}'")
            try:
                assistants = self.client.beta.assistants.list(limit=100)
                for assistant_data in assistants.data:
                    if assistant_data.name == self.assistant_name:
                        self.assistant_id = assistant_data.id
                        logger.info(f"Found existing assistant by name: '{self.assistant_name}' with ID: {self.assistant_id}")
                        return
            except OpenAIError as e:
                logger.error(f"Error listing assistants: {e}. Proceeding to create a new assistant.")

        if not self.assistant_id: # If still no ID, create a new assistant
            logger.info(f"No existing assistant found by name '{self.assistant_name}'. Creating a new one.")
            assistant_instructions = self.prompts.get('default_review_prompt', 'You are a helpful code review assistant. Please follow the provided JSON output format.')
            try:
                new_assistant = self.client.beta.assistants.create(
                    name=self.assistant_name,
                    instructions=assistant_instructions,
                    model=self.model,
                    tools=[] # No tools needed for this functionality yet
                )
                self.assistant_id = new_assistant.id
                logger.info(f"Successfully created new assistant '{self.assistant_name}' with ID: {self.assistant_id} and model: {self.model}")
            except OpenAIError as e:
                logger.error(f"Failed to create new assistant: {e}")
                raise Exception(f"Failed to initialize or create OpenAI assistant after all attempts: {e}") from e

        if not self.assistant_id:
            # This case should ideally be unreachable if creation was attempted and failed, as it would have raised.
            # But as a safeguard:
            logger.error("Failed to initialize or create OpenAI assistant after all attempts.")
            raise Exception("Failed to initialize or create OpenAI assistant.")

    def _wait_for_run_completion(self, thread_id: str, run_id: str, timeout_seconds: int = 300):
        start_time = time.time()
        logger.info(f"Waiting for run {run_id} in thread {thread_id} to complete...")
        while True:
            if time.time() - start_time > timeout_seconds:
                logger.error(f"Run {run_id} timed out after {timeout_seconds} seconds.")
                raise TimeoutError(f"Run completion timed out for run_id: {run_id}")

            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            except OpenAIError as e:
                logger.error(f"API error retrieving run {run_id}: {e}")
                # Depending on the error, might retry or raise immediately. For now, let loop retry.
                time.sleep(5) # Wait longer if API error
                continue

            logger.debug(f"Run {run_id} status: {run.status}")
            if run.status == 'completed':
                logger.info(f"Run {run_id} completed successfully.")
                return run
            elif run.status in ['failed', 'cancelled', 'expired']:
                logger.error(f"Run {run_id} ended with status: {run.status}. Last error: {run.last_error}")
                return run
            elif run.status == 'requires_action':
                logger.warning(f"Run {run_id} requires action, but no tools are defined. This is unexpected.")
                # For now, we don't handle actions, so this is effectively a terminal state for this bot.
                return run

            time.sleep(2) # Poll interval for in-progress states

    # Note: Old helper methods (_get_token_count, _get_prompt_template_tokens,
    # _generate_prompt, _split_patch) are removed in this change.

    def code_review(self, patch: str, model: str = None, temperature: float = None, top_p: float = None, max_tokens_for_response: int = None) -> dict:
        if not self.assistant_id:
            logger.error("Assistant not initialized. Cannot perform code review.")
            return {"lgtm": False, "review_comment": "Error: Assistant not initialized."}
        if not patch:
            return {"lgtm": True, "review_comment": ""}

        logger.info(f"Starting code review with Assistant ID: {self.assistant_id}")
        start_time = time.time()

        try:
            thread = self.client.beta.threads.create()
            logger.info(f"Created new thread: {thread.id}")

            json_instruction = self.prompts.get('json_format_requirement', 'Output in JSON: {"lgtm": bool, "review_comment": str}')
            message_content = (
                f"Please review this code patch:\n\n--- BEGIN PATCH ---\n{patch}\n--- END PATCH ---\n\n"
                f"Adhere strictly to this JSON output format:\n{json_instruction}"
            )

            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            logger.info(f"Added message to thread {thread.id}")

            run_params = {"assistant_id": self.assistant_id}
            if temperature is not None: run_params["temperature"] = temperature
            if top_p is not None: run_params["top_p"] = top_p
            if max_tokens_for_response is not None: run_params["max_completion_tokens"] = max_tokens_for_response
            if model is not None: run_params["model"] = model # Override assistant's model if provided

            run = self.client.beta.threads.runs.create(thread_id=thread.id, **run_params)
            logger.info(f"Created run {run.id} for thread {thread.id} with params: {run_params}")

            run_status = self._wait_for_run_completion(thread.id, run.id)

            if run_status.status == 'completed':
                messages_response = self.client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)

                if messages_response.data and messages_response.data[0].role == "assistant":
                    assistant_message = messages_response.data[0]
                    if assistant_message.content and isinstance(assistant_message.content, list) and assistant_message.content[0].type == "text":
                        response_text = assistant_message.content[0].text.value
                        logger.info(f"Raw response from assistant: {response_text}")
                        try:
                            if response_text.startswith('```json\n'): response_text = response_text[8:]
                            if response_text.endswith('\n```'): response_text = response_text[:-4]
                            elif response_text.endswith('```'): response_text = response_text[:-3]
                            review_data = json.loads(response_text)
                            return review_data
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON from assistant: {e}. Response: '{response_text}'")
                            return {"lgtm": False, "review_comment": f"Error: Could not parse AI response. Raw: {response_text}"}
                    else:
                        logger.error("Assistant message content not text or empty.")
                        return {"lgtm": False, "review_comment": "Error: Assistant response empty/not text."}
                else:
                    logger.error("No assistant messages found after completion.")
                    return {"lgtm": False, "review_comment": "Error: No response from assistant."}
            else:
                error_message = f"Review failed. Run status: {run_status.status}."
                if run_status.last_error:
                    error_message += f" Last Error: Code: {run_status.last_error.code}, Message: {run_status.last_error.message}"
                logger.error(error_message)
                return {"lgtm": False, "review_comment": error_message}

        except OpenAIError as e:
            logger.error(f"OpenAI API error during code review: {e}")
            return {"lgtm": False, "review_comment": f"Error: OpenAI API error - {str(e)}"}
        except TimeoutError as e:
            logger.error(f"Timeout during code review: {e}")
            return {"lgtm": False, "review_comment": f"Error: Code review timed out - {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during code review: {e}", exc_info=True)
            return {"lgtm": False, "review_comment": f"Error: An unexpected error occurred - {str(e)}"}
        finally:
            logger.info(f"Code review process finished in {time.time() - start_time:.2f} seconds.")

    # Definitions of _get_token_count, _get_prompt_template_tokens,
    # _generate_prompt, and _split_patch are removed from here.

    def _load_prompts(self) -> dict:
        # Attempt to load from MLflow Prompt Registry first
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_prompt_name = os.getenv("MLFLOW_PROMPT_NAME")
        mlflow_prompt_version = os.getenv("MLFLOW_PROMPT_VERSION") # Optional

        if mlflow_tracking_uri and mlflow_prompt_name:
            logger.info(f"Attempting to load prompts from MLflow Prompt Registry: Name='{mlflow_prompt_name}', Version='{mlflow_prompt_version or 'latest'}'")
            try:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

                prompt_uri = f"prompts:/{mlflow_prompt_name}"
                if mlflow_prompt_version:
                    prompt_uri += f"/{mlflow_prompt_version}"

                # Use mlflow.prompts.load_prompt as per the updated requirement
                prompt_yaml_str = mlflow.prompts.load_prompt(uri=prompt_uri)

                loaded_prompts = yaml.safe_load(prompt_yaml_str)

                # Validate the loaded prompts
                if isinstance(loaded_prompts, dict) and \
                   'default_review_prompt' in loaded_prompts and \
                   'json_format_requirement' in loaded_prompts:
                    logger.info("Successfully loaded and validated prompts from MLflow Prompt Registry.")
                    return loaded_prompts
                else:
                    logger.warning("Prompts loaded from MLflow Prompt Registry are invalid or missing expected keys. Proceeding to fallbacks.")

            except MlflowException as e:
                logger.error(f"MlflowException occurred while loading prompts from Prompt Registry: {e}")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML from MLflow Prompt Registry: {e}")
            # Removed the specific AttributeError check for `mlflow.llms.load_prompt`
            # as we are now directly using `mlflow.prompts.load_prompt`.
            # A general AttributeError or Exception will catch other issues.
            except AttributeError as e:
                 logger.error(f"An AttributeError occurred while loading prompts from MLflow Prompt Registry: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading prompts from MLflow Prompt Registry: {e}")
            logger.info("Falling back from MLflow Prompt Registry to other methods.")

        # Fallback to local prompts.yaml
        try:
            logger.info("Attempting to load prompts from local prompts.yaml file.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts.yaml')
            if os.path.exists(prompts_path): # Check if file exists before trying to open
                with open(prompts_path, 'r') as f:
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
            "json_format_requirement": "Provide your feedback in a strict JSON format with the following structure:\n{\n    \"lgtm\": boolean, // true if the code looks good to merge, false if there are concerns\n    \"review_comment\": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object."
        }

    # The _load_prompts method is now defined above code_review
    # This section will be replaced by the new code_review method.
    # The old helper methods (_get_token_count, _generate_prompt, etc.) will be removed in the next step.