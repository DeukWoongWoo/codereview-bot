import logging
import os
from typing import Any, Dict, Optional

import gitlab
from gitlab.exceptions import (GitlabCreateError,  # Added specific exceptions
                               GitlabGetError)
# To type hint gitlab.objects.ProjectMergeRequest more easily, though 'Any' can be used.
# from gitlab.objects import ProjectMergeRequest # This might be too specific for a general import

from chat import Chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitlabBot:
    """
    A bot that interacts with GitLab to review merge requests using an AI chat service.
    """
    def __init__(self, gitlab_url: str, gitlab_token: str, openai_api_key: str) -> None:
        """
        Initializes the GitlabBot.

        Args:
            gitlab_url: The URL of the GitLab instance.
            gitlab_token: The GitLab API token.
            openai_api_key: The OpenAI API key for the chat service.
        
        Raises:
            ValueError: If GITLAB_TOKEN is not provided.
        """
        self.gitlab_url: str = gitlab_url
        self.gitlab_token: str = gitlab_token
        self.openai_api_key: str = openai_api_key

        if not self.gitlab_token:
            raise ValueError("GITLAB_TOKEN is required")

        self.gl: gitlab.Gitlab = gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token)

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

    def _get_openai_params(self) -> Dict[str, Any]:
        """Reads and parses OpenAI related parameters from environment variables."""
        model: str = os.getenv("MODEL", "gpt-4o-mini")  # Default model in bot.py

        temperature_str: str = os.getenv("TEMPERATURE", "0.0")
        try:
            temperature: float = float(temperature_str)
        except ValueError:
            logger.warning(
                f"Invalid value for TEMPERATURE: '{temperature_str}'. Using default 0.0."
            )
            temperature = 0.0

        top_p_str: str = os.getenv("TOP_P", "1.0")
        try:
            top_p: float = float(top_p_str)
        except ValueError:
            logger.warning(
                f"Invalid value for TOP_P: '{top_p_str}'. Using default 1.0."
            )
            top_p = 1.0

        max_tokens_for_response_str: Optional[str] = os.getenv("MAX_TOKENS", None)
        max_tokens_for_response: Optional[int] = None
        if max_tokens_for_response_str is not None:
            try:
                max_tokens_for_response = int(max_tokens_for_response_str)
            except ValueError:
                logger.warning(
                    f"Invalid value for MAX_TOKENS: '{max_tokens_for_response_str}'. Using None."
                )
                # max_tokens_for_response remains None

        return {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens_for_response": max_tokens_for_response,
        }

    def _get_mr_patch(self, mr: Any) -> Optional[str]: # Using Any for mr object type
        """Retrieves and assembles the patch string from a merge request object."""
        try:
            changes = mr.changes()
            if not changes or not changes.get("changes"):
                logger.error(f"No changes found in merge request {mr.iid}.")
                return None

            patch_parts = []
            for change in changes["changes"]:
                if change.get("diff"):
                    patch_parts.append(
                        f"File: {change['new_path']}\n{change['diff']}\n\n"
                    )

            if not patch_parts:
                logger.info(f"No diff content found in changes for MR {mr.iid}.")
                return None  # Or consider empty string if that's more appropriate downstream

            return "".join(patch_parts)
        except (
            gitlab.exceptions.GitlabHttpError
        ) as e:  # More specific error for API issues
            logger.error(
                f"GitLab API error while fetching changes for MR {mr.iid}: {e}"
            )
            return None
        except Exception as e:  # Catch any other unexpected error
            logger.error(
                f"Unexpected error fetching or processing changes for MR {mr.iid}: {e}",
                exc_info=True,
            )
            return None

    def _post_review_comment(
        self, mr: Any, review: Dict[str, Any] # Using Any for mr, specific Dict for review
    ) -> None:
        """Formats and posts the review comment to the merge request."""
        try:
            comment_body: str = (
                "🤖 Code Review Bot\n\n"
                f"{'✅' if review.get('lgtm') else '⚠️'} LGTM: {review.get('lgtm', False)}\n\n"
                f"{review.get('review_comment', 'No review comment provided.')}"
            )
            mr.notes.create({"body": comment_body})
            logger.info(f"Successfully posted review comment to MR {mr.iid}.")
        except GitlabCreateError as e:
            logger.error(
                f"GitLab API error (GitlabCreateError) while posting review comment to MR {mr.iid}: {e}"
            )
            # Attempt to post a simplified error message to GitLab
            try:
                mr.notes.create(
                    {
                        "body": f"❌ An error occurred while trying to post the full review comment to MR {mr.iid}.\nOriginal error: {e}"
                    }
                )
                logger.info(f"Posted simplified error notification to MR {mr.iid}.")
            except Exception as inner_e:  # Catch potential error during fallback posting
                logger.error(
                    f"Failed to post even the simplified error notification to MR {mr.iid}: {inner_e}",
                    exc_info=True,
                )
        except Exception as e:  # Catch any other unexpected error during comment formatting or posting
            logger.error(
                f"Unexpected error while formatting or posting review comment to MR {mr.iid}: {e}",
                exc_info=True,
            )
            # Optionally, try to post a generic error message if the above specific GitlabCreateError didn't catch it.
            try:
                mr.notes.create(
                    {
                        "body": f"❌ An critical unexpected error occurred while trying to process and post the review for MR {mr.iid}."
                    }
                )
            except Exception:
                pass  # If even this fails, we've already logged the main error.

    def handle_merge_request(self, project_id: int, merge_request_iid: int) -> None:
        """
        Orchestrates the handling of a GitLab merge request event.

        Retrieves MR details, gets the patch, requests a code review,
        and posts the review comments back to GitLab.
        """
        logger.info(
            f"Handling merge request event for project_id: {project_id}, mr_iid: {merge_request_iid}"
        )
        project: Any # Using Any for project object type
        mr: Any # Using Any for MR object type
        try:
            project = self.gl.projects.get(project_id)
            mr = project.mergerequests.get(merge_request_iid)
            logger.info(f"Successfully retrieved MR: {mr.title} (ID: {mr.iid})")
        except GitlabGetError as e:
            logger.error(
                f"GitLab API error (GitlabGetError) while retrieving project or MR: {e}. ProjectID: {project_id}, MR_IID: {merge_request_iid}."
            )
            return
        except Exception as e:  # Catch any other unexpected error during MR retrieval
            logger.error(
                f"Unexpected error while retrieving project or MR: {e}. ProjectID: {project_id}, MR_IID: {merge_request_iid}.",
                exc_info=True,
            )
            return

        chat_service: Optional[Chat] = self.load_chat()
        if not chat_service:
            # load_chat() already logs the error, so just return.
            return

        patch_str: Optional[str] = self._get_mr_patch(mr)
        if patch_str is None:
            # _get_mr_patch() already logs the error or info about no diffs.
            return
        if not patch_str.strip():  # If patch is whitespace only
            logger.info(
                f"Patch for MR {mr.iid} is empty or whitespace only. Skipping review."
            )
            return

        openai_params: Dict[str, Any] = self._get_openai_params()

        try:
            logger.info(
                f"Requesting code review for MR {mr.iid} with model {openai_params['model']}."
            )
            review_content: Dict[str, Any] = chat_service.code_review(
                patch=patch_str, # Use the renamed variable
                model=openai_params["model"],
                temperature=openai_params["temperature"],
                top_p=openai_params["top_p"],
                max_tokens_for_response=openai_params["max_tokens_for_response"],
            )
        except Exception as e:  # Catch errors from chat.code_review() itself
            logger.error(
                f"Code review generation failed for MR {mr.iid}: {e}", exc_info=True
            )
            # Try to post an error message to GitLab
            error_review: Dict[str, Any] = {
                "lgtm": False,
                "review_comment": f"❌ Code review generation failed due to an internal error: {e}",
            }
            self._post_review_comment(mr, error_review)
            return

        if review_content:
            logger.info(
                f"Code review received for MR {mr.iid}. LGTM: {review_content.get('lgtm')}"
            )
            self._post_review_comment(mr, review_content)
        else:
            # This case should ideally be handled by chat.code_review returning an error dict.
            logger.error(
                f"No review content received from chat module for MR {mr.iid}."
            )
            error_review = { # Explicit type for clarity
                "lgtm": False,
                "review_comment": "❌ Code review failed: No content received from the review service.",
            }
            self._post_review_comment(mr, error_review)


def main() -> None:
    required_env_vars = {
        "CI_SERVER_URL": os.getenv("CI_SERVER_URL"),
        "CI_PROJECT_ID": os.getenv("CI_PROJECT_ID"),
        "CI_MERGE_REQUEST_IID": os.getenv("CI_MERGE_REQUEST_IID"),
        "GITLAB_TOKEN": os.getenv("GITLAB_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    GITLAB_URL = required_env_vars["CI_SERVER_URL"]
    PROJECT_ID = required_env_vars["CI_PROJECT_ID"]
    MR_IID = required_env_vars["CI_MERGE_REQUEST_IID"]
    GITLAB_TOKEN = required_env_vars["GITLAB_TOKEN"]
    OPENAI_API_KEY = required_env_vars["OPENAI_API_KEY"]

    bot = GitlabBot(GITLAB_URL, GITLAB_TOKEN, OPENAI_API_KEY)
    bot.handle_merge_request(PROJECT_ID, MR_IID)


if __name__ == "__main__":
    main()
