# GitLab Code Review Bot

An automated code review bot for GitLab merge requests using OpenAI's GPT models. The bot analyzes merge request changes and provides intelligent code review comments.

## Features

- Automated code review for GitLab merge requests
- Integration with OpenAI's GPT models
- Customizable review prompts
- Detailed feedback including LGTM status and improvement suggestions

## Prerequisites

- Python 3.11 or higher
- GitLab account with API access
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) package manager

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DeukWoongWoo/codereview-bot.git
   cd codereview-bot
   ```

2. Create and configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` file with your configuration:
   ```
   CI_SERVER_URL=<your-gitlab-url>
   CI_PROJECT_ID=<project-id>
   CI_MERGE_REQUEST_IID=<merge-request-iid>
   GITLAB_TOKEN=<your-gitlab-token>
   OPENAI_API_KEY=<your-openai-api-key>
   ```

3. Install dependencies using uv:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install .
   ```

## Running the Bot

### Using uv with environment variables

```bash
uv run --env-file=.env python bot.py
```

### Optional Environment Variables

- `MODEL`: OpenAI model to use (default: 'gpt-4o-mini')
- `TEMPERATURE`: Model temperature (default: 0.0)
- `TOP_P`: Top P sampling (default: 1.0)
- `MAX_TOKENS`: Maximum tokens for response (optional)
- `PROMPT`: Custom review prompt (optional)

## Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t gitlab-codereview-bot .
   ```

2. Run the container:
   ```bash
   docker run --env-file .env gitlab-codereview-bot
   ```

## Development

### Customizing Review Prompts

Edit `prompts.json` to customize the review prompts and format requirements:

```json
{
    "default_review_prompt": "Your custom prompt here",
    "json_format_requirement": "Format requirements here"
}
```

## License

MIT
