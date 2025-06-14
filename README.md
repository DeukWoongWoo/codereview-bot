# GitLab Code Review Bot

An automated code review bot for GitLab merge requests, now powered by the **OpenAI Assistants API**. The bot analyzes merge request changes and provides intelligent code review comments using a persistent Assistant, managing context within conversation Threads.

## Features

- Automated code review for GitLab merge requests.
- Utilizes OpenAI Assistants API for more robust and stateful interactions.
- Customizable Assistant instructions via MLflow Prompt Registry or local `prompts.yaml`.
- Structured JSON output for reviews, including LGTM status and detailed comments.

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
-   `OPENAI_BASE_URL`: (Optional) Custom base URL for OpenAI API calls (defaults to `https://api.openai.com/v1`).

### Assistant Configuration:

-   `OPENAI_ASSISTANT_ID`: (Optional) The ID of a pre-configured OpenAI Assistant. If provided, the bot will use this Assistant. If not, it will attempt to find an existing one by name or create a new one.
-   `OPENAI_ASSISTANT_NAME`: (Optional) Name for the OpenAI Assistant. Used when finding an existing assistant or creating a new one if `OPENAI_ASSISTANT_ID` is not provided. Defaults to `GitLabCodeReviewAssistant`.
-   `MODEL`: The OpenAI model to be used by the Assistant (e.g., `gpt-4o`, `gpt-4-turbo-preview`). Defaults to `gpt-4o` if not set. This model is used when a new Assistant is created, or can optionally override the Assistant's model on a per-run basis if specified.

### Run Parameters (for each review):

-   `TEMPERATURE`: (Optional) Controls randomness in the model's output. Lower is more deterministic (default: `0.0` passed to Assistant Run).
-   `TOP_P`: (Optional) Nucleus sampling parameter (default: `1.0` passed to Assistant Run).
-   `MAX_TOKENS`: (Optional) Maximum number of tokens for the Assistant's response. This is passed as `max_completion_tokens` to the Assistant Run, controlling the output length.

### Prompt Management (for Assistant Instructions and Formatting):

The bot defines its behavior and output requirements using prompts, which are loaded with the following priority:
1.  **MLflow Prompt Registry** (if `MLFLOW_TRACKING_URI` and `MLFLOW_PROMPT_NAME` are set)
2.  Local `prompts.yaml` file
3.  Hardcoded defaults (as a final fallback)

The loaded prompts provide:
-   `default_review_prompt`: This text is used as the primary "instructions" when creating or configuring the OpenAI Assistant. It defines the assistant's role, expertise, and general guidelines for the review.
-   `json_format_requirement`: This text is appended to each user message (containing the code patch) sent to the Assistant. It instructs the Assistant to format its entire response as a single, valid JSON object with specific keys (`lgtm`, `review_comment`).

-   **`MLFLOW_TRACKING_URI`**: URI of your MLflow tracking server (e.g., `http://localhost:5000`).
-   **`MLFLOW_PROMPT_NAME`**: The name of the prompt in the MLflow Prompt Registry (e.g., `code_review_bot_prompts`). The content of this registry prompt should be a YAML string (see example below).
-   **`MLFLOW_PROMPT_VERSION`**: (Optional) The version of the prompt in the MLflow Prompt Registry. If not provided, the latest version is fetched.

**Content for MLflow Prompt Registry / `prompts.yaml`:**
The prompt source (MLflow or `prompts.yaml`) should define a YAML structure that provides `default_review_prompt` and `json_format_requirement`. Example:
```yaml
# This entire block is the content of the MLflow Registry prompt string or prompts.yaml
default_review_prompt: |
  You are an expert AI code review assistant. Your task is to meticulously review code changes (diffs/patches) from GitLab merge requests. Focus exclusively on the new changes (added or modified code). Use deleted code or surrounding code only as context to understand the intent or impact of the new changes, but do not include them in the review feedback unless they directly affect the new changes.

  Your review should identify:
  - Potential bugs (e.g., logic errors, race conditions, edge cases not handled).
  # ... (rest of the detailed instructions as defined previously) ...
  If the provided patch is empty, malformed, or appears to contain malicious code, explicitly state this and decline to provide a detailed review.

json_format_requirement: |
  Always provide your complete feedback as a single, valid JSON object adhering to the following structure:
  {
    "lgtm": boolean, // true if the new changes are generally good to merge without critical issues; false if there are significant concerns or the patch is invalid.
    "review_comment": string // Your detailed review comments. Use Markdown syntax within this string for formatting (e.g., code blocks, lists). For each issue or improvement, include a clear description and a concrete suggestion. If no issues are found, use a concise statement like "No significant issues found in the new changes." Must be non-empty and provide actionable feedback. If the patch is invalid or malicious, set "lgtm" to false and explain why in "review_comment".
  }
  Ensure the entire response is a single, valid JSON object and nothing else.
```
(Note: The example above is truncated for brevity in this plan; the full content from the previous step should be used in the actual file.)

If MLflow is not used or fails, the bot looks for a `prompts.yaml` file in its root directory.

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

If not using MLflow for prompt management, you can customize the Assistant's instructions and JSON format requirements by editing the `prompts.yaml` file in the root directory. See the example structure under "Prompt Management".

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
