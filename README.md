# GitLab Code Review Bot

An automated code review bot for GitLab merge requests using OpenAI's GPT models. The bot analyzes merge request changes and provides intelligent code review comments.

## Features

- Automated code review for GitLab merge requests.
- Integration with OpenAI's GPT models.
- Customizable review prompts with multiple sourcing options (MLflow Prompt Registry, local YAML, defaults).
- Handles large diffs by intelligently chunking them for review.
- Detailed feedback including LGTM status and improvement suggestions.

## Prerequisites

- Python 3.11 or higher
- GitLab account with API access
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) package manager (for local setup and development)

## Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/DeukWoongWoo/codereview-bot.git # Replace with your repo URL if forked
    cd codereview-bot
    ```

2.  **Create and configure environment variables**:
    Copy the example environment file and edit it with your specific configuration:
    ```bash
    cp .env.example .env
    nano .env # Or your preferred editor
    ```
    See the "Configuration (Environment Variables)" section below for details on all variables.

3.  **Install dependencies using uv**:
    ```bash
    uv venv # Create a virtual environment
    source .venv/bin/activate # Activate the environment (on Linux/macOS)
    # On Windows, use: .venv\Scripts\activate
    uv pip install -e . # Install the bot and its dependencies in editable mode
    ```

## Running the Bot

Once configured, you can run the bot using:

```bash
uv run python bot.py
```
This command will use the environment variables defined in your `.env` file.

## Configuration (Environment Variables)

The bot is configured using environment variables. These can be placed in a `.env` file in the project root.

### Core Configuration:

-   `CI_SERVER_URL`: Your GitLab instance URL (e.g., `https://gitlab.com`).
-   `CI_PROJECT_ID`: The ID of the GitLab project the bot should operate on.
-   `CI_MERGE_REQUEST_IID`: The IID (internal ID) of the merge request to be reviewed.
-   `GITLAB_TOKEN`: Your GitLab personal access token with `api` scope.
-   `OPENAI_API_KEY`: Your OpenAI API key.

### Model and Review Parameters:

-   `MODEL`: OpenAI model to use for reviews (default: `gpt-4o-mini`).
-   `TEMPERATURE`: Controls randomness in the model's output. Lower is more deterministic (default: `0.0`).
-   `TOP_P`: Nucleus sampling parameter (default: `1.0`).
-   `MAX_TOKENS`: Maximum number of tokens to generate in the review response from the OpenAI model (optional). Note: This controls the *output* length. For input length, see Prompt Chunking.
-   `PROMPT`: Overrides the `default_review_prompt` from the loaded prompts (see Prompt Management). Useful for quick tests or specific overrides without changing prompt files.

### Prompt Management (MLflow Prompt Registry):

Prompts define how the bot interacts with the OpenAI model. They are loaded with the following priority:
1.  **MLflow Prompt Registry** (if `MLFLOW_TRACKING_URI` and `MLFLOW_PROMPT_NAME` are set)
2.  Local `prompts.yaml` file
3.  Hardcoded defaults (as a final fallback)

-   **`MLFLOW_TRACKING_URI`**: URI of your MLflow tracking server (e.g., `http://localhost:5000`).
-   **`MLFLOW_PROMPT_NAME`**: The name of the prompt in the MLflow Prompt Registry (e.g., `code_review_bot_prompts`).
-   **`MLFLOW_PROMPT_VERSION`**: (Optional) The version of the prompt in the MLflow Prompt Registry. If not provided, the latest version is fetched.

**Content for MLflow Prompt Registry:**
The prompt stored in the MLflow Prompt Registry should be a single YAML string that can be parsed into a dictionary containing `default_review_prompt` and `json_format_requirement` keys. Example:
```yaml
# This entire block should be stored as the string value for the prompt in MLflow.
default_review_prompt: |
  As an AI assistant specializing in code review, please analyze the following code patch.
  Focus on identifying potential bugs, security vulnerabilities, areas for performance improvement,
  and adherence to best practices. Provide specific, actionable suggestions.
json_format_requirement: |
  Structure your feedback as a valid JSON object with two main keys:
  1. "lgtm": A boolean value (true if the patch is generally good to merge, false if there are significant concerns).
  2. "review_comment": A string containing your detailed review. You can use Markdown syntax within this string.
  Example:
  {
    "lgtm": true,
    "review_comment": "### Overall Assessment:\nLooks good overall. Just a few minor suggestions..."
  }
```
If MLflow is not used or fails, the bot looks for a `prompts.yaml` file in its root directory (see "Customizing Review Prompts (`prompts.yaml`)" under Development).

### Handling Large Diffs (Prompt Chunking):

To manage very large code changes that might exceed the OpenAI model's context window, the bot can automatically split the diff into smaller chunks. Each chunk is reviewed separately, and the feedback is aggregated.

-   **`MAX_TOKENS_OVERRIDE`**: (Optional) Allows you to set a custom maximum token limit for the selected OpenAI model, overriding the bot's predefined defaults. This limit is used to decide if chunking is necessary.
-   **`MAX_REVIEW_CHUNKS`**: (Optional) The maximum number of chunks a diff will be split into. If splitting would result in more chunks, the review is aborted with an error message (default: `10`). This prevents excessive API calls for extremely large diffs.
-   **`PROMPT_RESPONSE_BUFFER`**: (Internal, not typically user-set) An internal buffer of tokens reserved for the model's response and formatting, when calculating chunk sizes.

## Docker Setup

1.  **Build the Docker image**:
    ```bash
    docker build -t gitlab-codereview-bot .
    ```

2.  **Run the container**:
    Ensure your `.env` file is populated with the necessary environment variables.
    ```bash
    docker run --env-file .env gitlab-codereview-bot
    ```

## Development

### Customizing Review Prompts (`prompts.yaml`)

If not using MLflow for prompt management (i.e., `MLFLOW_TRACKING_URI` or `MLFLOW_PROMPT_NAME` are not set, or MLflow loading fails), you can customize review prompts by editing the `prompts.yaml` file in the root directory.

The basic structure is:

```yaml
default_review_prompt: |
  As an AI assistant specializing in code review, please analyze the following code patch.
  Focus on identifying potential bugs, security vulnerabilities, areas for performance improvement,
  and adherence to best practices. Provide specific, actionable suggestions.

json_format_requirement: |
  Structure your feedback as a valid JSON object with two main keys:
  1. "lgtm": A boolean value (true if the patch is generally good to merge, false if there are significant concerns).
  2. "review_comment": A string containing your detailed review. You can use Markdown syntax within this string.
  Example:
  {
    "lgtm": true,
    "review_comment": "### Overall Assessment:\nLooks good overall. Just a few minor suggestions..."
  }
```

### Running Tests

Unit tests are located in the `tests` directory. To run them:

1.  Ensure you are in the project's root directory and the virtual environment is activated.
2.  Install test dependencies (if any, though typically `unittest` is part of Python's standard library and other dependencies are covered by `uv pip install .`):
    ```bash
    # No specific test-only dependencies for now beyond what's in pyproject.toml
    ```
3.  Run the tests using Python's `unittest` module:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```

## License

MIT
