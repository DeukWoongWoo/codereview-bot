from openai import OpenAI
import os
import time
import yaml
import json
import logging
import mlflow
from mlflow.exceptions import MlflowException
import tiktoken # For token counting and chunking
# yaml is imported lower down. os, time, json are also used.

logger = logging.getLogger(__name__)

# Default model token limits (conservative estimates)
MODEL_TOKEN_LIMITS = {
    'gpt-4o-mini': 128000,
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096, # Older versions might be 4k, newer 16k. Check specific variant.
    'text-davinci-003': 4000, # Example, if using older completion models
}
DEFAULT_ENCODER = "cl100k_base" # A common encoder

# Max chunks to split a review into, to prevent excessive API calls
DEFAULT_MAX_REVIEW_CHUNKS = 10
# Buffer tokens to leave for the model's response and JSON structure
PROMPT_RESPONSE_BUFFER = 1024 # Increased buffer

class Chat:
    def __init__(self, api_key: str):
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        self.prompts = self._load_prompts()

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

    def _get_token_count(self, text: str, model_name: str) -> int:
        """Helper to count tokens using tiktoken."""
        if not text:
            return 0
        try:
            # Attempt to get the encoding for the specific model
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to a default encoder if the model is not found
            logger.warning(f"Model '{model_name}' not found by tiktoken. Using default encoder '{DEFAULT_ENCODER}'.")
            encoding = tiktoken.get_encoding(DEFAULT_ENCODER)

        return len(encoding.encode(text))

    def _get_prompt_template_tokens(self, model_name: str) -> int:
        """Calculates token count for the base prompt template (without the patch)."""
        # This simulates _generate_prompt with an empty patch to count template tokens
        # Ensure PROMPT and json_format_requirement are from self.prompts
        user_prompt_template = os.getenv('PROMPT', self.prompts.get('default_review_prompt', ''))
        json_format_template = self.prompts.get('json_format_requirement', '')

        # Add markers for clarity, though they add a few tokens
        template_text = f"{user_prompt_template}\n{json_format_template}\n--- BEGIN PATCH ---\n--- END PATCH ---"
        return self._get_token_count(template_text, model_name)

    def _generate_prompt(self, patch_content: str) -> str: # Renamed 'patch' to 'patch_content'
        user_prompt = os.getenv('PROMPT', self.prompts['default_review_prompt'])
        json_format_requirement = self.prompts['json_format_requirement']
        # Markers to clearly delineate the patch for token calculation and review focus
        return f"{user_prompt}\n{json_format_requirement}\n--- BEGIN PATCH ---\n{patch_content}\n--- END PATCH ---"

    def _split_patch(self, full_patch: str, max_tokens_per_chunk: int, model: str) -> list[str]:
        """Splits a large patch into chunks, trying to preserve file boundaries and diff context."""
        chunks = []
        if not full_patch:
            return []

        # Split by file diffs first: "diff --git a/... b/..."
        # Regex to split by "diff --git " but keep the delimiter
        file_diffs = []
        current_diff = ""
        for line in full_patch.splitlines(True): # Keep newlines
            if line.startswith("diff --git ") and current_diff:
                file_diffs.append(current_diff.strip())
                current_diff = line
            else:
                current_diff += line
        if current_diff: # Add the last diff
            file_diffs.append(current_diff.strip())

        if not file_diffs: # If no "diff --git " found, treat as one large block
            file_diffs = [full_patch]

        for file_diff in file_diffs:
            file_diff_token_count = self._get_token_count(file_diff, model)

            if file_diff_token_count <= max_tokens_per_chunk:
                chunks.append(file_diff)
            else:
                # File diff is too large, split by lines (simplistic, could be by hunks)
                logger.info(f"A file diff is too large ({file_diff_token_count} tokens), splitting by lines.")
                lines = file_diff.splitlines(True)
                current_chunk_lines = []
                current_chunk_tokens = 0
                header_lines = [line for line in lines if line.startswith("--- a/") or line.startswith("+++ b/") or line.startswith("diff --git ") or line.startswith("index ")]

                for line in lines:
                    # Always include header lines in each sub-chunk of this file_diff for context
                    if line in header_lines and line not in current_chunk_lines:
                        line_tokens = self._get_token_count("".join(header_lines), model) # Approx
                    else:
                        line_tokens = self._get_token_count(line, model)

                    if current_chunk_tokens + line_tokens > max_tokens_per_chunk and current_chunk_lines:
                        # Prepend necessary headers if not already there (basic context)
                        final_chunk_lines = list(dict.fromkeys(header_lines + current_chunk_lines)) # Keep order, unique
                        chunks.append("".join(final_chunk_lines))
                        current_chunk_lines = []
                        current_chunk_tokens = self._get_token_count("".join(header_lines), model) if header_lines else 0

                    current_chunk_lines.append(line)
                    current_chunk_tokens += line_tokens

                if current_chunk_lines: # Add remaining part of the large diff
                    final_chunk_lines = list(dict.fromkeys(header_lines + current_chunk_lines))
                    chunks.append("".join(final_chunk_lines))
        
        # Post-process to ensure no chunk is truly empty if original patch was not
        return [chunk for chunk in chunks if chunk.strip()]


    def code_review(self, patch: str, model: str = 'gpt-4o-mini', temperature: float = 0.0, top_p: float = 1.0, max_tokens_for_response: int = None) -> dict: # Renamed max_tokens to max_tokens_for_response
        if not patch:
            return {"lgtm": True, "review_comment": ""}

        start_time = time.time()

        # Determine model's token limit
        model_specific_limit = MODEL_TOKEN_LIMITS.get(model, MODEL_TOKEN_LIMITS['gpt-4o-mini']) # Default if model not in map
        max_total_tokens = int(os.getenv('MAX_TOKENS_OVERRIDE', model_specific_limit))

        # Calculate tokens for the prompt template (without the patch)
        template_tokens = self._get_prompt_template_tokens(model)

        # Calculate tokens for the patch itself
        patch_tokens = self._get_token_count(patch, model)

        # Total tokens for the initial full prompt (template + patch)
        # The _generate_prompt adds markers, already accounted for in template_tokens if it uses _generate_prompt("")
        # Or, more accurately:
        base_user_prompt = os.getenv('PROMPT', self.prompts.get('default_review_prompt', ''))
        base_json_req = self.prompts.get('json_format_requirement', '')
        full_prompt_text_for_calc = f"{base_user_prompt}\n{base_json_req}\n--- BEGIN PATCH ---\n{patch}\n--- END PATCH ---"
        current_prompt_total_tokens = self._get_token_count(full_prompt_text_for_calc, model)

        logger.info(f"Model: {model}, Max Total Tokens: {max_total_tokens}")
        logger.info(f"Template Tokens: {template_tokens}, Patch Tokens: {patch_tokens}, Current Prompt Total Tokens: {current_prompt_total_tokens}")

        # Effective limit for the patch content itself, considering template and response buffer
        # This is the target for each chunk if splitting is needed.
        max_patch_tokens_for_single_request = max_total_tokens - template_tokens - PROMPT_RESPONSE_BUFFER

        if current_prompt_total_tokens <= max_total_tokens:
            logger.info("Patch fits within token limits. Processing as a single request.")
            try:
                # Use the `patch` variable directly as `_generate_prompt` expects the content part
                prompt_to_send = self._generate_prompt(patch)

                response = self.openai.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_to_send}],
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens_for_response # This is for the response, not total prompt
                )
                content = response.choices[0].message.content
                if content.startswith('```json\n'): content = content[8:]
                if content.endswith('```'): content = content[:-3]
                return json.loads(content)
            except Exception as e:
                logger.error(f"Error during single-chunk code review: {e}")
                raise e
            finally:
                logger.info(f"Code review (single chunk) took {time.time() - start_time:.2f} seconds")
        else:
            logger.info(f"Patch exceeds token limits ({current_prompt_total_tokens} > {max_total_tokens}). Splitting into chunks.")
            
            max_review_chunks = int(os.getenv('MAX_REVIEW_CHUNKS', DEFAULT_MAX_REVIEW_CHUNKS))
            # max_tokens_per_patch_chunk is the target for the content of the patch in each chunk
            max_tokens_per_patch_chunk = max_patch_tokens_for_single_request
            logger.info(f"Target max tokens for each patch chunk content: {max_tokens_per_patch_chunk}")

            patch_chunks = self._split_patch(patch, max_tokens_per_patch_chunk, model)

            if not patch_chunks:
                logger.error("Patch splitting resulted in no chunks. Cannot proceed.")
                return {"lgtm": False, "review_comment": "Error: Diff is too large or complex, and splitting failed."}

            if len(patch_chunks) > max_review_chunks:
                logger.error(f"Splitting resulted in {len(patch_chunks)} chunks, exceeding MAX_REVIEW_CHUNKS ({max_review_chunks}).")
                return {"lgtm": False, "review_comment": f"Error: Diff is too large, resulting in {len(patch_chunks)} chunks (max {max_review_chunks}). Please review manually or submit smaller changes."}

            aggregated_reviews_content = []
            overall_lgtm = True

            logger.info(f"Processing patch in {len(patch_chunks)} chunks.")

            for i, chunk_content in enumerate(patch_chunks):
                chunk_start_time = time.time()
                logger.info(f"Reviewing chunk {i+1}/{len(patch_chunks)}...")
                # Generate prompt for this specific chunk
                prompt_chunk = self._generate_prompt(chunk_content)

                try:
                    response = self.openai.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_chunk}],
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens_for_response
                    )
                    content = response.choices[0].message.content
                    if content.startswith('```json\n'): content = content[8:]
                    if content.endswith('```'): content = content[:-3]

                    review_data = json.loads(content)

                    # Add a header to each chunk's review for clarity
                    chunk_header = f"--- Review for Chunk {i+1}/{len(patch_chunks)} ---\n"
                    aggregated_reviews_content.append(chunk_header + review_data.get('review_comment', 'No comment for this chunk.'))
                    overall_lgtm = overall_lgtm and review_data.get('lgtm', True)
                    logger.info(f"Chunk {i+1} review completed in {time.time() - chunk_start_time:.2f}s. LGTM: {review_data.get('lgtm')}")

                except Exception as e:
                    logger.error(f"Error during code review for chunk {i+1}: {e}")
                    aggregated_reviews_content.append(f"--- Error reviewing Chunk {i+1}/{len(patch_chunks)} ---\n{str(e)}")
                    overall_lgtm = False # Mark as not LGTM if any chunk fails

            combined_review_string = "\n\n".join(aggregated_reviews_content)
            logger.info(f"Aggregated code review (chunked) took {time.time() - start_time:.2f} seconds. Overall LGTM: {overall_lgtm}")
            return {"lgtm": overall_lgtm, "review_comment": combined_review_string}
        # The original finally block is removed as each path (single vs chunked) now has its own timing log.
        # If a general finally is needed, it can be added here.