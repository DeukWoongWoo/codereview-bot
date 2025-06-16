import logging
import gitlab
from jinja2 import Template
from openai import OpenAI
from pydantic import Field
from pydantic import BaseSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MergeRequestError(Exception):
    """Exception for merge request errors."""
    pass

class AppConfig(BaseSettings):
    # GitLab settings
    ci_server_url: str = Field(alias="CI_SERVER_URL")
    ci_project_id: str = Field(alias="CI_PROJECT_ID")
    ci_merge_request_iid: str = Field(alias="CI_MERGE_REQUEST_IID")
    gitlab_token: str = Field(alias="GITLAB_TOKEN")
    
    # OpenAI settings
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="MODEL")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_p: float = Field(default=1.0, alias="TOP_P")
    max_tokens: int | None = Field(default=None, alias="MAX_TOKENS")
    


def load_prompt_template(description, comments, changes):
    """Load and render the prompt template."""
    try:
        with open('prompt.j2', 'r', encoding='utf-8') as f:
            template = Template(f.read())
        
        return template.render(
            description=description,
            comments='\n'.join(comments),
            changes=changes
        )
    except FileNotFoundError:
        logger.error("prompt.j2 template file not found")
        raise
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        raise

def get_merge_request_data(config: AppConfig):
    """Fetch merge request data from GitLab."""
    try:
        gl = gitlab.Gitlab(config.ci_server_url, private_token=config.gitlab_token)
        project = gl.projects.get(config.ci_project_id)
        mr = project.mergerequests.get(config.ci_merge_request_iid)
        return mr
    except Exception as e:
        logger.error(f"Failed to fetch merge request: {e}")
        raise MergeRequestError(f"Could not retrieve merge request {config.ci_merge_request_iid}")

def extract_user_comments(mr):
    """Extract only user comments from the merge request."""
    comments = []
    for note in mr.notes.list():
        if getattr(note, 'system', False):
            continue
        if note.author and note.author.get('name') == 'GitLab':
            continue
        comments.append(note.body)
    return comments

def build_patch_content(mr):
    """Assemble changes as patch content."""
    changes = mr.changes()
    if not changes.get('changes'):
        raise MergeRequestError("No changes found in merge request")
    
    patch_parts = []
    for change in changes['changes']:
        if change.get('diff'):
            patch_parts.append(f"File: {change['new_path']}")
            patch_parts.append(change['diff'])
            patch_parts.append("")
    return '\n'.join(patch_parts)

def generate_ai_response(config: AppConfig, prompt: str):
    """Generate a response using the OpenAI API."""
    client = OpenAI(api_key=config.openai_api_key)
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

def main():
    try:
        config = AppConfig()
        
        mr = get_merge_request_data(config)
        comments = extract_user_comments(mr)
        patch = build_patch_content(mr)
        prompt = load_prompt_template(mr.description, comments, patch)
        ai_response = generate_ai_response(config, prompt)
        
        mr.notes.create({'body': ai_response})
        logger.info("AI analysis comment added successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()