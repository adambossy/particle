import asyncio
import json
import logging
import os
import pprint
import random
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import instructor
import litellm
from dotenv import load_dotenv
from fireworks.client import Fireworks

from particle.call_graph_analyzer import is_test_file
from particle.llm_results_parser import LLMResultsParser
from particle.utils import (
    get_assistant_message_from_tool_call,
    get_user_message_from_tool_call,
    should_use_function_calling,
)

from .file_manager import FileManager

load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# litellm._turn_on_debug()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("particle.llm_translator")


def translate_code(translated_code: str, error: str | None = None) -> str:
    """
    Translate the code and return the updated translated source.
    """
    pass


# Use this for gpt-4o
# gpt_4o_translate_code_tool = {
#     "type": "function",
#     "function": litellm.utils.function_to_dict(translate_code),
# }


deepseek_client = Fireworks(
    api_key=os.getenv("FIREWORKS_AI_API_KEY"),
)


gpt_4o_translate_code_tool = {
    "type": "function",
    "function": {
        "name": "translate_code",
        "description": "Translate the code and return the updated translated source.",
        "parameters": {
            "type": "object",
            "properties": {
                "translated_code": {
                    "type": "string",
                    "description": "The source code that has been translated from Python to Go.",
                },
                "error": {
                    "type": "string",
                    "description": "The error message, if any, encountered during translation.",
                },
            },
            "required": ["translated_code"],
        },
    },
}


# NOTE (adam) This is Claude-specific and will break for other models
claude_translate_code_tool = {
    "name": "translate_code",
    "description": "Translate the code from Python to Go and return the updated translated source.",
    "input_schema": {
        "type": "object",
        "properties": {
            "translated_code": {
                "type": "string",
                "description": "The source code that has been translated from Python to Go.",
            },
            "error": {
                "type": "string",
                "description": "The error message, if any, encountered during translation.",
            },
        },
        "required": ["translated_code"],
    },
}


translate_code_tool_table = {
    "deepseek/deepseek-coder": gpt_4o_translate_code_tool,
    "deepseek/deepseek-chat": gpt_4o_translate_code_tool,
    "fireworks_ai/accounts/fireworks/models/deepseek-v3": gpt_4o_translate_code_tool,
    "gpt-4o-2024-08-06": gpt_4o_translate_code_tool,
    "o3-mini": gpt_4o_translate_code_tool,
    "anthropic/claude-3-5-sonnet-20241022": claude_translate_code_tool,
    "anthropic/claude-3-7-sonnet-20250219": claude_translate_code_tool,
    "gemini/gemini-2.0-flash": gpt_4o_translate_code_tool,
    "gemini/gemini-1.5-pro": gpt_4o_translate_code_tool,
}


# Rate limit configuration by model
rate_limit_config = {
    # OpenAI models
    "gpt-4o-2024-08-06": {
        "rpm_limit": 10000,
        "tpm_limit": 30000000,
    },
    # Anthropic models
    "anthropic/claude-3-5-sonnet-20241022": {
        "rpm_limit": 100,
        "ipm_limit": 80000,
        "opm_limit": 32000,
    },
    "anthropic/claude-3-7-sonnet-20250219": {
        "rpm_limit": 100,
        "ipm_limit": 80000,
        "opm_limit": 32000,
    },
    # Gemini models
    "gemini/gemini-2.0-flash": {
        "rpm_limit": 2000,
        "tpm_limit": 4000000,
    },
    "gemini/gemini-1.5-pro": {
        "rpm_limit": 1000,
        "tpm_limit": 4000000,
    },
}

# Default rate limits for models not in the lookup table
DEFAULT_RATE_LIMITS = {
    "rpm_limit": 100,  # Default to Claude's RPM limit
    "ipm_limit": 80000,  # Default to Claude's IPM limit
    "opm_limit": 32000,  # Default to Claude's OPM limit
}


class RateLimitTracker:
    """Tracks rate limit information for API calls."""

    def __init__(self, model: str) -> None:
        # Store the model name
        self.model: str = model

        # Get rate limits for the specified model or use defaults
        model_limits = rate_limit_config.get(model, DEFAULT_RATE_LIMITS)

        # Set rate limits based on model configuration
        self.rpm_limit: int = model_limits.get(
            "rpm_limit", DEFAULT_RATE_LIMITS["rpm_limit"]
        )

        # Check if the model uses TPM or IPM/OPM
        if "tpm_limit" in model_limits:
            self.tpm_limit: int = model_limits["tpm_limit"]
            self.ipm_limit: Optional[int] = None
            self.opm_limit: Optional[int] = None
            self.uses_tpm: bool = True
        else:
            self.tpm_limit: Optional[int] = None
            self.ipm_limit: int = model_limits.get(
                "ipm_limit", DEFAULT_RATE_LIMITS["ipm_limit"]
            )
            self.opm_limit: int = model_limits.get(
                "opm_limit", DEFAULT_RATE_LIMITS["opm_limit"]
            )
            self.uses_tpm: bool = False

        # Current usage
        self.rpm_current: int = 0
        self.ipm_current: int = 0
        self.opm_current: int = 0
        self.tpm_current: int = 0  # Combined token usage for TPM models

        # Last reset time
        self.last_reset_time: datetime = datetime.now()

        # Throttling state
        self.is_throttled: bool = False
        self.throttle_until: Optional[datetime] = None

    def update_from_headers(self, headers: Dict[str, Any]) -> None:
        """Update rate limit information from response headers."""
        # Extract rate limit information from headers
        # Format depends on the API provider

        # For Anthropic, headers might look like:
        # x-ratelimit-limit-requests: 100
        # x-ratelimit-remaining-requests: 95
        # x-ratelimit-limit-tokens: 32000
        # x-ratelimit-remaining-tokens: 30000

        pprint.pformat(headers)

        if "x-ratelimit-limit-requests" in headers:
            self.rpm_limit = int(
                headers.get("x-ratelimit-limit-requests", self.rpm_limit)
            )

        if self.uses_tpm:
            if "x-ratelimit-limit-tokens" in headers:
                self.tpm_limit = int(
                    headers.get("x-ratelimit-limit-tokens", self.tpm_limit)
                )

            if "x-ratelimit-remaining-tokens" in headers:
                remaining = int(headers.get("x-ratelimit-remaining-tokens", 0))
                self.tpm_current = self.tpm_limit - remaining
        else:
            if "x-ratelimit-limit-input-tokens" in headers:
                self.ipm_limit = int(
                    headers.get("x-ratelimit-limit-input-tokens", self.ipm_limit)
                )

            if "x-ratelimit-limit-output-tokens" in headers:
                self.opm_limit = int(
                    headers.get("x-ratelimit-limit-output-tokens", self.opm_limit)
                )

            if "x-ratelimit-remaining-input-tokens" in headers:
                remaining = int(headers.get("x-ratelimit-remaining-input-tokens", 0))
                self.ipm_current = self.ipm_limit - remaining

            if "x-ratelimit-remaining-output-tokens" in headers:
                remaining = int(headers.get("x-ratelimit-remaining-output-tokens", 0))
                self.opm_current = self.opm_limit - remaining

        if "x-ratelimit-remaining-requests" in headers:
            remaining = int(headers.get("x-ratelimit-remaining-requests", 0))
            self.rpm_current = self.rpm_limit - remaining

        # Update from usage in response
        self.log_current_usage()

    def update_from_usage(self, usage: Dict[str, Any]) -> None:
        """Update rate limit information from response usage data."""
        if usage:
            # Update token counts if available
            if self.uses_tpm:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                self.tpm_current += prompt_tokens + completion_tokens
            else:
                if "prompt_tokens" in usage:
                    self.ipm_current += usage.get("prompt_tokens", 0)
                if "completion_tokens" in usage:
                    self.opm_current += usage.get("completion_tokens", 0)

            # Increment request count
            self.rpm_current += 1

            self.log_current_usage()

    def should_throttle(self) -> Tuple[bool, float]:
        """Determine if we should throttle requests based on current usage."""
        # If we're already throttled, check if we can resume
        if self.is_throttled and self.throttle_until:
            if datetime.now() >= self.throttle_until:
                self.is_throttled = False
                self.throttle_until = None
                logger.info("Resuming requests after throttling period")
            else:
                wait_time = (self.throttle_until - datetime.now()).total_seconds()
                return True, wait_time

        # Check if we're approaching limits (90% of limit)
        approaching_rpm_limit = self.rpm_current >= 0.9 * self.rpm_limit

        if self.uses_tpm:
            approaching_token_limit = self.tpm_current >= 0.9 * self.tpm_limit
        else:
            approaching_ipm_limit = self.ipm_current >= 0.9 * self.ipm_limit
            approaching_opm_limit = self.opm_current >= 0.9 * self.opm_limit
            approaching_token_limit = approaching_ipm_limit or approaching_opm_limit

        # If we're approaching any limit, throttle
        if approaching_rpm_limit or approaching_token_limit:
            # Calculate time until next minute (when limits reset)
            now = datetime.now()
            next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            wait_time = (next_minute - now).total_seconds()

            self.is_throttled = True
            self.throttle_until = next_minute

            if approaching_rpm_limit:
                logger.warning(
                    f"Approaching RPM limit ({self.rpm_current}/{self.rpm_limit}). Throttling for {wait_time:.2f}s"
                )

            if self.uses_tpm:
                if approaching_token_limit:
                    logger.warning(
                        f"Approaching TPM limit ({self.tpm_current}/{self.tpm_limit}). Throttling for {wait_time:.2f}s"
                    )
            else:
                if approaching_ipm_limit:
                    logger.warning(
                        f"Approaching IPM limit ({self.ipm_current}/{self.ipm_limit}). Throttling for {wait_time:.2f}s"
                    )
                if approaching_opm_limit:
                    logger.warning(
                        f"Approaching OPM limit ({self.opm_current}/{self.opm_limit}). Throttling for {wait_time:.2f}s"
                    )

            return True, wait_time

        return False, 0

    def reset_if_needed(self) -> None:
        """Reset counters if a minute has passed since last reset."""
        now = datetime.now()
        if (now - self.last_reset_time).total_seconds() >= 60:
            if self.uses_tpm:
                logger.info(
                    f"Resetting rate limit counters. Previous usage: RPM={self.rpm_current}/{self.rpm_limit}, "
                    f"TPM={self.tpm_current}/{self.tpm_limit}"
                )
                self.tpm_current = 0
            else:
                logger.info(
                    f"Resetting rate limit counters. Previous usage: RPM={self.rpm_current}/{self.rpm_limit}, "
                    f"IPM={self.ipm_current}/{self.ipm_limit}, OPM={self.opm_current}/{self.opm_limit}"
                )
                self.ipm_current = 0
                self.opm_current = 0

            self.rpm_current = 0
            self.last_reset_time = now

    def log_current_usage(self) -> None:
        """Log current usage metrics."""
        if self.uses_tpm:
            logger.info(
                f"Current API usage: RPM={self.rpm_current}/{self.rpm_limit}, "
                f"TPM={self.tpm_current}/{self.tpm_limit}"
            )
        else:
            logger.info(
                f"Current API usage: RPM={self.rpm_current}/{self.rpm_limit}, "
                f"IPM={self.ipm_current}/{self.ipm_limit}, OPM={self.opm_current}/{self.opm_limit}"
            )


def get_source_code_from_translations_dict(
    translated_code_by_file: dict[Path, str]
) -> str:
    """Return the source code from a dictionary of file paths to code."""
    for file_name, translated_code in translated_code_by_file.items():
        file_path = Path(file_name)
        if not is_test_file(file_path):
            return translated_code
    raise ValueError("No source code file found in the provided dictionary.")


class LLMTranslator:
    def __init__(
        self,
        model: str,
        file_manager: FileManager,
        metadata: dict[str, Any] = {},
    ):
        super().__init__()

        self.model = model
        self.file_manager = file_manager
        self.metadata = metadata

        self.llm_results_parser = LLMResultsParser()

        # Determine if the model supports function calling
        self.uses_function_calling: bool = should_use_function_calling(model)

        self.client = instructor.from_litellm(litellm.acompletion)

        # Initialize rate limit tracker
        self.rate_tracker = RateLimitTracker(model)

        # Retry configuration
        self.max_retries = 5
        self.base_retry_delay = 1.0  # seconds

        self.system_prompt = """You are an expert AI assistant focused on translating Python code to Go.
When translating Python code to Go:
1. Produce idiomatic Go code that follows Go best practices and maintains test compatibility
2. Ensure all variables and functions are properly typed
3. Use appropriate Go data structures and patterns
4. Preserve the original functionality and logic exactly as specified by tests
5. Include necessary imports
6. Convert Python docstrings to Go comments
7. Handle Python-specific features appropriately:
   - Convert list comprehensions to loops or slices
   - Replace Python dictionaries with appropriate Go maps
   - Convert Python classes to Go structs and methods
   - Implement Python-style exception handling using Go error handling
8. When tests are provided:
   - Convert Python test frameworks (pytest, unittest) to Go testing package
   - Maintain test coverage and assertions
   - Convert Python fixtures to Go test helpers
   - Preserve test descriptions and documentation
   - Handle test data and mocks appropriately
   - Ensure test setup and teardown are properly translated
9. Return the complete, properly formatted Go code with tests (if applicable)
10. Return the full relative file path in the new repo that it belongs to, followed by the translated code.

Remember: The translated code must pass all provided tests after conversion.
"""

        self.one_shot_user_message = """Translate this Python code and its tests to Go.
For each code snippet, prepend the translation with a comment containing the full relative file path
in the new repo that it belongs to, followed by the translated code. Use the file mappings that are
provided to map the file path to the new repo.

The package name should be the same as the enclosing directory of the file.

File Mappings:
py_project/src/implementation.py -> go_project/src/implementation.go
py_project/tests/test_implementation.py -> go_project/src/implementation_test.go

# py_project/src/implementation.py
def filter_and_transform(items):
    '''
    Filter out negative numbers and transform the remaining ones.
    Returns a list of strings in the format "num: <value>"
    '''
    return [f"num: {x}" for x in items if x >= 0]

# py_project/tests/test_implementation.py
import pytest
from . import implementation

def test_filter_and_transform():
    # Test with mixed positive and negative numbers
    assert filter_and_transform([-1, 0, 1, -2, 2]) == ["num: 0", "num: 1", "num: 2"]
    
    # Test with empty list
    assert filter_and_transform([]) == []
    
    # Test with all negative numbers
    assert filter_and_transform([-1, -2, -3]) == []"""

        self.one_shot_assistant_message = """// go_project/src/implementation.go
package src

import "fmt"

// filterAndTransform filters out negative numbers and transforms the remaining ones.
// Returns a slice of strings in the format "num: <value>"
func filterAndTransform(items []int) []string {
    result := make([]string, 0, len(items))
    for _, x := range items {
        if x >= 0 {
            result = append(result, fmt.Sprintf("num: %d", x))
        }
    }
    return result
}

// go_project/src/implementation_test.go
package src

import {
    "reflect"
    "testing"
}

func TestFilterAndTransform(t *testing.T) {
    tests := []struct {
        name     string
        input    []int
        expected []string
    }{
        {
            name:     "mixed positive and negative numbers",
            input:    []int{-1, 0, 1, -2, 2},
            expected: []string{"num: 0", "num: 1", "num: 2"},
        },
        {
            name:     "empty list",
            input:    []int{},
            expected: []string{},
        },
        {
            name:     "all negative numbers",
            input:    []int{-1, -2, -3},
            expected: []string{},
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := filterAndTransform(tt.input)
            if !reflect.DeepEqual(result, tt.expected) {
                t.Errorf("filterAndTransform() = %v, want %v", result, tt.expected)
            }
        })
    }
}"""

        self.clear_messages()

    def compose_prompt(
        self,
        code_snippets_by_file: dict[str, list[str]],
        special_instructions: str | None = None,
    ) -> str:
        special_instructions = (
            (special_instructions + "\n\n") if special_instructions else ""
        )

        composed_prompt = f"""Translate this Python code and its tests to Go.
For each code snippet, prepend the translation with a comment containing the full relative file path
in the new repo that it belongs to, followed by the translated code. Use the file mappings that are
provided to map the file path to the new repo.

The package name should be the same as the enclosing directory of the file.
{special_instructions}
File Mappings:\n"""
        for py_filename in code_snippets_by_file.keys():
            py_file_path = Path(py_filename)
            go_file_path = self.file_manager.make_go_file_path(py_file_path)
            composed_prompt += (
                f"{py_file_path.as_posix()} -> {go_file_path.as_posix()}\n"
            )

        composed_prompt += "\n"
        for filename, code_snippets in code_snippets_by_file.items():
            composed_prompt += f"# {filename}\n"
            for code_snippet in code_snippets:
                composed_prompt += f"{code_snippet}\n\n"
            composed_prompt += "\n"

        return composed_prompt

    async def get_completion(self) -> Dict[str, Any]:
        """Get completion based on the model type."""
        if not self.uses_function_calling:
            return await self.get_plain_text_completion()

        # Check if the model is from Anthropic
        is_anthropic_model = self.model.startswith("anthropic/")

        if is_anthropic_model:
            # Add extra parameter to get response headers
            completion = await litellm.acompletion(
                messages=self.messages,
                model=self.model,
                tools=[translate_code_tool_table[self.model]],
                tool_choice={
                    "type": "function",
                    "function": {"name": "translate_code"},
                },
                temperature=1.0,
                # return_response_headers=True,  # Get response headers
            )

            # Extract headers and update rate tracker
            # FIXME (adam) update_from_headers never gets called so I don't think
            # this is the right check
            if hasattr(completion, "_response_ms") and hasattr(
                completion._response_ms, "headers"
            ):
                self.rate_tracker.update_from_headers(completion._response_ms.headers)

            # Update from usage data
            if hasattr(completion, "usage"):
                self.rate_tracker.update_from_usage(completion.usage)
        else:
            completion = await litellm.acompletion(
                messages=self.messages,
                model=self.model,
                tools=[translate_code_tool_table[self.model]],
                tool_choice={
                    "type": "function",
                    "function": {"name": "translate_code"},
                },
                temperature=1.0,
            )

            # Update from usage data for other models
            if hasattr(completion, "usage"):
                self.rate_tracker.update_from_usage(completion.usage)

        # Save the completion for potential retry operations
        self.last_completion = completion
        return completion

    async def get_plain_text_completion(self) -> str:
        """Get completion for models that don't use function calling."""
        # Use the appropriate client based on the model
        # if self.model.startswith("fireworks_ai/"):
        #     # acompletion_func = deepseek_client.chat.completions.acreate
        #     # For fireworks models, we need to use the full path after "fireworks_ai/"
        #     if self.model == "fireworks_ai/accounts/fireworks/models/deepseek-v3":
        #         model_name = "accounts/fireworks/models/deepseek-v3"
        #     else:
        #         # Extract the model name from the full path
        #         model_name = self.model.replace("fireworks_ai/", "")
        # else:
        #     # For other non-function calling models, use litellm
        #     acompletion_func = litellm.acompletion
        #     model_name = self.model

        acompletion_func = litellm.acompletion

        completion = await acompletion_func(
            messages=self.messages,
            model=self.model,
            max_tokens=20000,
            temperature=1.0,
            stream=False,
        )

        # Save the completion for potential retry operations
        self.last_completion = completion

        # Update rate tracker if usage data is available
        if hasattr(completion, "usage"):
            self.rate_tracker.update_from_usage(completion.usage)

        # Extract code from between fences
        content = completion.choices[0].message.content

        if self.model in ["gemini/gemini-1.5-pro", "gemini/gemini-2.0-flash"]:
            translated_code_by_file = self.llm_results_parser.parse_translations(
                content
            )
            return get_source_code_from_translations_dict(translated_code_by_file)

        if self.model == "fireworks_ai/accounts/fireworks/models/deepseek-v3":
            # Find code blocks between triple backticks
            code_blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)\n```", content)

            translated_code = None
            if code_blocks:
                for block in code_blocks:
                    lines = block.splitlines()
                    if lines and lines[0].startswith("//"):
                        # Extract the filename from the comment
                        file_comment = lines[0]
                        filename = file_comment[
                            2:
                        ].strip()  # Remove "//" and strip whitespace
                        file_path = Path(filename)

                        # Check if it's a test file
                        if not is_test_file(file_path):
                            # Omit the first line and keep the rest of the code
                            translated_code = "\n".join(lines[1:])
                if not translated_code:
                    logger.warning(
                        f"Couldn't extract source code from blocks, returning full content (metadata: {self.metadata})"
                    )
                    logger.warning(content)
            else:
                logger.warning(
                    f"No code blocks found in response, returning full content (metadata: {self.metadata})"
                )
                logger.warning(content)

            # If no code blocks found, return the entire content as a fallback
            return translated_code or content

    async def completion(self) -> None:
        # Check if we need to reset rate limit counters
        self.rate_tracker.reset_if_needed()

        # Check if we should throttle
        should_throttle, wait_time = self.rate_tracker.should_throttle()
        if should_throttle:
            logger.info(f"Throttling API requests. Waiting for {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        # Implement retry logic with exponential backoff
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                if not self.uses_function_calling:
                    translated_code = await self.get_plain_text_completion()
                    # Add the assistant message to the conversation history
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": self.last_completion.choices[0].message.content,
                        }
                    )
                    return translated_code

                completion = await self.get_completion()

                # For models using tools
                self.messages.append(
                    get_assistant_message_from_tool_call(self.model, completion)
                )

                tool_call = completion.choices[0].message.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)
                return arguments["translated_code"]

            except litellm.exceptions.RateLimitError as e:
                retry_count += 1

                # Extract rate limit information from error if possible
                error_msg = str(e)
                logger.warning(f"Rate limit error encountered: {error_msg}")

                if retry_count > self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded. Giving up."
                    )
                    raise

                # Calculate backoff time (exponential with jitter)
                backoff_time = self.base_retry_delay * (2**retry_count) + (
                    0.1 * random.random()
                )

                # If error message contains information about when to retry, use that
                if "try again later" in error_msg.lower():
                    # Try to extract a wait time from the error message
                    # This is a simplistic approach - in practice you might need more sophisticated parsing
                    backoff_time = (
                        60  # Default to 60 seconds if we can't parse a specific time
                    )

                logger.info(
                    f"Retrying in {backoff_time:.2f} seconds (attempt {retry_count}/{self.max_retries})"
                )
                await asyncio.sleep(backoff_time)

            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(traceback.format_exc())

                retry_count += 1

                if retry_count > self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded. Giving up."
                    )
                    raise

                # Calculate backoff time (exponential with jitter)
                backoff_time = self.base_retry_delay * (2**retry_count) + (
                    0.1 * random.random()
                )
                logger.info(
                    f"Retrying in {backoff_time:.2f} seconds (attempt {retry_count}/{self.max_retries})"
                )
                await asyncio.sleep(backoff_time)

    async def translate(
        self,
        code_snippets_by_file: dict[str, list[str]],
        special_instructions: str | None = None,
    ) -> str:
        composed_prompt = self.compose_prompt(
            code_snippets_by_file,
            special_instructions,
        )
        self.messages.append({"role": "user", "content": composed_prompt})

        translate_code = await self.completion()
        return translate_code

    async def retry(self, last_test_output: str, test_code: str | None = None) -> str:
        self.messages.append(
            get_user_message_from_tool_call(
                self.model,
                self.last_completion,
                last_test_output,
                test_code,
                is_error=True,
            )
        )

        return await self.completion()

    def initialize_messages(self) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.one_shot_user_message},
            {"role": "assistant", "content": self.one_shot_assistant_message},
        ]

    def clear_messages(self) -> None:
        self.initialize_messages()


DEFAULT_SOURCE_AND_FILES = {
    "src/process_data.py": [
        """def process_data(data):
    '''Process the input data'''
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""
    ],
    "tests/test_process_data.py": [
        """def test_process_data():
    # Test with a list of positive numbers
    assert process_data([1, 2, 3]) == [2, 4, 6]
    
    # Test with an empty list
    assert process_data([]) == []
    
    # Test with a list containing zero
    assert process_data([0, 1, 2]) == [0, 2, 4]
"""
    ],
}


@click.command()
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    help="Path to the project directory to analyze",
)
@click.option(
    "--model",
    type=click.Choice(["gpt-4o", "claude"]),
    default="claude",
    help="The LLM model to use",
)
def cli(project_path: str, model: str) -> None:
    """CLI tool for translating Python code to Go."""
    from benchmark.main import SUPPORTED_MODELS

    llm = SUPPORTED_MODELS[model]()
    file_manager = FileManager(Path(project_path))
    llm_translator = LLMTranslator(llm, file_manager)

    for filename, code_snippets in DEFAULT_SOURCE_AND_FILES.items():
        click.echo(f"# {filename}\n")
        for code_snippet in code_snippets:
            click.echo(f"{code_snippet}\n")
        click.echo("\n")

    click.echo("\nTranslated Go code:")
    translated_go_code = llm_translator.translate(DEFAULT_SOURCE_AND_FILES)

    click.echo("\nResult:")
    click.echo(translated_go_code)


if __name__ == "__main__":
    cli()
