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

            # Gather user comments on the merge request
            comments = ""
            try:
                notes = mr.notes.list(all=True)
                user_notes = [note.body for note in notes if not getattr(note, "system", False)]
                comments = "\n".join(user_notes)
            except Exception as e:
                logger.error(f"Failed to fetch merge request comments: {e}")

            # Get code review from ChatGPT
            try:
                model = os.getenv('MODEL', 'gpt-4o-mini')
                temperature = float(os.getenv('TEMPERATURE', 0.0))
                top_p = float(os.getenv('TOP_P', 1.0))
                max_tokens = os.getenv('MAX_TOKENS', None)

                review_context = {
                    "description": mr.description,
                    "comments": comments,
                    "patch": patch,
                    "created_at": mr.created_at,
                }

                chunk_size = int(os.getenv("CHUNK_SIZE", "0")) or None
                review = chat.code_review_agent(
                    review_context,
                    model,
                    temperature,
                    top_p,
                    max_tokens,
                    chunk_size,
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
                logger.error(f"Error during code review: {e}")
                mr.notes.create({
                    'body': f"❌ An error occurred while reviewing this merge request.\n error message: {e}\n\n original review comment : {review['review_comment']}"
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
