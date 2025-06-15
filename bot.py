import os
import logging
import gitlab
from jinja2 import Template
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt(description, comments, changes):
    with open('prompt.j2', 'r', encoding='utf-8') as f:
        prompt_template = Template(f.read())
    
    return prompt_template.render(
        description=description,
        comments='\n'.join(comments),
        changes=changes
    )

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
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please ensure they are set in your .env file.")
        raise ValueError("Check the logs for details.")

        
    GITLAB_URL = required_env_vars["CI_SERVER_URL"]
    PROJECT_ID = required_env_vars["CI_PROJECT_ID"]
    MR_IID = required_env_vars["CI_MERGE_REQUEST_IID"]
    GITLAB_TOKEN = required_env_vars["GITLAB_TOKEN"]
    OPENAI_API_KEY = required_env_vars["OPENAI_API_KEY"]

    # Get MR
    gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    mr = project.mergerequests.get(MR_IID)

    # Collect all non-system comments (notes)
    mr_comments = []
    for note in mr.notes.list():
        # Exclude system notes (system=True) or author name is 'GitLab'
        if getattr(note, 'system', False):
            continue
        if note.author and note.author.get('name') == 'GitLab':
            continue
        mr_comments.append(f"{note.body}")

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

    prompt = load_prompt(mr.description, mr_comments, changes)
    print(prompt)

    openai = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1"
    )

    model = os.getenv("MODEL", "gpt-4o-mini")
    temperature = float(os.getenv('TEMPERATURE', 0.0))
    top_p = float(os.getenv('TOP_P', 1.0))
    max_tokens = os.getenv('MAX_TOKENS', None)

    response = openai.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    content = response.choices[0].message.content
    mr.notes.create({'body': content})


if __name__ == "__main__":
    main()
