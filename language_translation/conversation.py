import json
from typing import Any, Callable

import instructor
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

load_dotenv()


class Conversation:
    def __init__(self):
        """
        Initialize Conversation with a LiteLLM-compatible model.

        Args:
            model: Model identifier (e.g. "anthropic/claude-3-sonnet-20240229")
        """

        self.models = [
            "anthropic/claude-3-sonnet-20240229",
            "gpt-4o-2024-08-06",
            "gemini/gemini-1.5-pro-latest",
        ]
        self.client = instructor.from_litellm(completion)

    def primary_model(self) -> str:
        return self.models[0]

    def validated_completion(
        self,
        messages: list[dict] | str,
        validator: Callable[[Any], bool],
        attempts_per_model: int = 1,
        response_model: BaseModel | None = None,
    ) -> Any:
        """
        Get a completion and validate it, falling back to other models if validation fails.

        Args:
            messages: The input messages or prompt string
            validator: Function that takes completion result and returns bool
            attempts_per_model: Maximum attempts per model
            tools: Optional list of function definitions for function calling
            tool_choice: Optional specification of which function to call

        Returns:
            The validated completion result

        Raises:
            Exception: If validation fails for all models after max attempts
        """
        for model in self.models:
            for _ in range(attempts_per_model):
                print(f"Making completion call with {model}")
                result = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    response_model=response_model,
                )
                if validator(result):
                    return result

        raise Exception(
            f"Failed to get valid completion after {attempts_per_model} attempts per model"
        )
