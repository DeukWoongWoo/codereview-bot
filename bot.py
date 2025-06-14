import os
import logging
from typing import Optional
import gitlab
from chat import Chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitlabBot:
    def __init__(self, gitlab_url: str, gitlab_token: str, openai_api_key: str):
        self.gitlab_url = gitlab_url
        self.gitlab_token = gitlab_token
        self.openai_api_key = openai_api_key
        
        if not self.gitlab_token:
            raise ValueError("GITLAB_TOKEN is required")
            
        self.gl = gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token)
    
    def load_chat(self) -> Optional[Chat]:
        """Initialize Chat instance with OpenAI API key."""
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY is not set")
            return None
        
        try:
            return Chat(self.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Chat: {e}")
            return None

    def handle_merge_request(self, project_id: int, merge_request_iid: int):
        """Handle GitLab merge request events."""
        try:
            project = self.gl.projects.get(project_id)
            mr = project.mergerequests.get(merge_request_iid)
            
            # Initialize chat
            chat = self.load_chat()
            if not chat:
                logger.error("Chat initialization failed")
                return
            
            # Get the changes (diff/patch)
            changes = mr.changes()
            if not changes.get('changes'):
                logger.error("No changes found in merge request")
                return
            
            # Combine all changes into a single patch
            patch = ""
            for change in changes['changes']:
                if change.get('diff'):
                    patch += f"File: {change['new_path']}\n"
                    patch += f"{change['diff']}\n\n"
            
            # Get code review from ChatGPT
            try:
                model = os.getenv('MODEL', 'gpt-4o-mini')
                temperature = float(os.getenv('TEMPERATURE', 0.0))
                top_p = float(os.getenv('TOP_P', 1.0))
                # max_tokens in chat.code_review is now max_tokens_for_response
                max_tokens_for_response = os.getenv('MAX_TOKENS', None)
                if max_tokens_for_response is not None:
                    try:
                        max_tokens_for_response = int(max_tokens_for_response)
                    except ValueError:
                        logger.warning(f"Invalid value for MAX_TOKENS: '{max_tokens_for_response}'. Using None.")
                        max_tokens_for_response = None
                
                # chat.code_review now expects the patch string directly as the first argument.
                # Other metadata like description, created_at are not directly used by chat.py's prompt generation anymore.
                # If they are needed, chat.py's _generate_prompt would need to be adapted.
                review = chat.code_review(
                    patch=patch,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens_for_response=max_tokens_for_response
                )
            except Exception as e:
                logger.error(f"Error during code review: {e}")
                return

            try:
                # Create comment with review
                comment = (
                    "🤖 Code Review Bot\n\n"
                    f"{'✅' if review['lgtm'] else '⚠️'} LGTM: {review['lgtm']}\n\n"
                    f"{review['review_comment']}"
                )
                
                mr.notes.create({'body': comment})
            except Exception as e:
                logger.error(f"Error posting review comment to GitLab: {e}")
                # Attempt to post a simplified error message to GitLab if the original comment fails
                try:
                    mr.notes.create({
                        'body': f"❌ An error occurred while trying to post the full review comment.\nError: {e}"
                    })
                except Exception as inner_e:
                    logger.error(f"Failed to post even a simplified error comment to GitLab: {inner_e}")
                })

        except Exception as e:
            logger.error(f"Error handling merge request: {e}")

def main():
    required_env_vars = {
        "CI_SERVER_URL": os.getenv("CI_SERVER_URL"),
        "CI_PROJECT_ID": os.getenv("CI_PROJECT_ID"),
        "CI_MERGE_REQUEST_IID": os.getenv("CI_MERGE_REQUEST_IID"),
        "GITLAB_TOKEN": os.getenv("GITLAB_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    GITLAB_URL = required_env_vars["CI_SERVER_URL"]
    PROJECT_ID = required_env_vars["CI_PROJECT_ID"]
    MR_IID = required_env_vars["CI_MERGE_REQUEST_IID"]
    GITLAB_TOKEN = required_env_vars["GITLAB_TOKEN"]
    OPENAI_API_KEY = required_env_vars["OPENAI_API_KEY"]

    bot = GitlabBot(GITLAB_URL, GITLAB_TOKEN, OPENAI_API_KEY)
    bot.handle_merge_request(PROJECT_ID, MR_IID)

if __name__ == "__main__":
    main()
