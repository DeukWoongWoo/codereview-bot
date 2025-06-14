from openai import OpenAI
import os
import time
import yaml
import mlflow
import json
import logging
from typing import List, Dict

from utils import split_patch

logger = logging.getLogger(__name__)


class Chat:
    def __init__(self, api_key: str):
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )
        self.prompts = self._load_prompts()
        self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        if not self.assistant_id:
            try:
                instructions = (
                    self.prompts["default_review_prompt"]
                    + "\n"
                    + self.prompts["json_format_requirement"]
                )
                assistant = self.openai.beta.assistants.create(
                    name="Code Review Agent",
                    instructions=instructions,
                    model=os.getenv("MODEL", "gpt-4o-mini"),
                )
                self.assistant_id = assistant.id
            except Exception as e:
                logger.error(f"Failed to create assistant: {e}")
                self.assistant_id = None


    def _load_prompts(self) -> dict:
        default_name = os.getenv("DEFAULT_PROMPT_NAME", "default_review_prompt")
        json_name = os.getenv("JSON_FORMAT_PROMPT_NAME", "json_format_requirement")
        try:
            default_prompt = mlflow.load_prompt(default_name)
            json_prompt = mlflow.load_prompt(json_name)
            return {
                "default_review_prompt": default_prompt.template,
                "json_format_requirement": json_prompt.template,
            }
        except Exception as e:
            logger.error(f"Failed to load prompts from MLflow: {e}")
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                prompts_path = os.path.join(current_dir, "prompts.yaml")
                with open(prompts_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading prompts.yaml: {e}")
                return {
                    "default_review_prompt": "Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.",
                    "json_format_requirement": "Provide your feedback in a strict JSON format with the following structure:\n{\n    \"lgtm\": boolean, // true if the code looks good to merge, false if there are concerns\n    \"review_comment\": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object.",
                }
    def _generate_prompt(self, review_context: dict) -> str:
        user_prompt = os.getenv('PROMPT', self.prompts['default_review_prompt'])
        json_format_requirement = self.prompts['json_format_requirement']

        description = review_context.get("description", "")
        comments = review_context.get("comments", "")
        patch = review_context.get("patch", "")

        context = (
            f"Description:\n{description}\n\n"
            f"Comments:\n{comments}\n\n"
            f"Patch:\n{patch}"
        )

        return f"{user_prompt}\n{json_format_requirement}:\n{context}"

    def _assistant_review(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict:
        run = self.openai.beta.threads.create_and_run_poll(
            assistant_id=self.assistant_id,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            thread={"messages": [{"role": "user", "content": prompt}]},
        )

        messages = self.openai.beta.threads.messages.list(run.thread_id)
        content = messages.data[0].content[0].text.value
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].lstrip()
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content)

    @staticmethod
    def _combine_results(results: List[dict]) -> dict:
        if not results:
            return {"lgtm": True, "review_comment": ""}

        lgtm = all(r.get("lgtm", False) for r in results)
        comment = "\n".join(r.get("review_comment", "") for r in results if r.get("review_comment"))
        return {"lgtm": lgtm, "review_comment": comment}

    def code_review(self, review_context: dict, model: str = 'gpt-4o-mini', temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = None) -> dict:
        if not review_context.get("patch"):
            return {
                "lgtm": True,
                "review_comment": ""
            }

        start_time = time.time()
        try:
            prompt = self._generate_prompt(review_context)
            response = self.openai.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content
            # Remove markdown code block if present
            if content.startswith('```json\n'):
                content = content[8:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error during code review: {e}")
            raise e
        finally:
            print(f"Code review took {time.time() - start_time:.2f} seconds")

    def code_review_agent(
        self,
        review_context: dict,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        chunk_size: int | None = None,
    ) -> dict:
        """Perform code review using the OpenAI Assistant API.

        If ``chunk_size`` is provided and the patch exceeds this size,
        the patch will be split and reviewed in multiple rounds.
        """

        if not review_context.get("patch"):
            return {"lgtm": True, "review_comment": ""}

        if not self.assistant_id:
            raise RuntimeError("Assistant is not initialized")

        start_time = time.time()
        try:
            patch = review_context.get("patch", "")
            chunks = (
                split_patch(patch, chunk_size or len(patch))
                if patch
                else [""]
            )

            results: List[Dict] = []
            for chunk in chunks:
                ctx = review_context.copy()
                ctx["patch"] = chunk
                prompt = self._generate_prompt(ctx)
                res = self._assistant_review(
                    prompt, model, temperature, top_p, max_tokens
                )
                results.append(res)

            return self._combine_results(results)
        except Exception as e:
            logger.error(f"Error during agent code review: {e}")
            raise e
        finally:
            print(f"Agent review took {time.time() - start_time:.2f} seconds")

