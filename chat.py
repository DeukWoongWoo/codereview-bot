from openai import OpenAI
import os
import time
import json

class Chat:
    def __init__(self, api_key: str):
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts.json')
            with open(prompts_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading prompts.json: {e}")
            return {
                "default_review_prompt": "Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.",
                "json_format_requirement": "Provide your feedback in a strict JSON format with the following structure:\n{\n    \"lgtm\": boolean, // true if the code looks good to merge, false if there are concerns\n    \"review_comment\": string // Your detailed review comments. You can use markdown syntax in this string, but the overall response must be a valid JSON\n}\nEnsure your response is a valid JSON object."
            }

    def _generate_prompt(self, patch: str) -> str:
        user_prompt = os.getenv('PROMPT', self.prompts['default_review_prompt'])
        json_format_requirement = self.prompts['json_format_requirement']
        return f"{user_prompt}{json_format_requirement} :\n{patch}"

    def code_review(self, patch: str, model: str = 'gpt-4o-mini', temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = None) -> dict:
        if not patch:
            return {
                "lgtm": True,
                "review_comment": ""
            }
        
        start_time = time.time()
        try:
            prompt = self._generate_prompt(patch)
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
        finally:
            print(f"Code review took {time.time() - start_time:.2f} seconds")
 