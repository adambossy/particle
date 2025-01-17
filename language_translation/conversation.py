import json
from typing import Any, Callable

from dotenv import load_dotenv
from litellm import completion

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

    def primary_model(self) -> str:
        return self.models[0]

    def validated_completion(
        self,
        messages: list[dict] | str,
        validator: Callable[[Any], bool],
        attempts_per_model: int = 1,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
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
                result = self.completion(
                    messages,
                    model,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                if validator(result):
                    return result

        raise Exception(
            f"Failed to get valid completion after {attempts_per_model} attempts per model"
        )

    def completion(
        self,
        messages: list[dict] | str,
        model: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            messages: The input messages or prompt string
            model: Optional model override
            tools: Optional list of function definitions for function calling
            tool_choice: Optional specification of which function to call

        Returns:
            The completion response content or function call result
        """
        if not model:
            model = self.primary_model()

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if hasattr(self, "system_prompt"):
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        completion_args = {"model": model, "messages": messages}
        if tools:
            completion_args["tools"] = tools
        if tool_choice:
            completion_args["tool_choice"] = tool_choice

        response = completion(**completion_args)

        # Handle function calling response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)

        return response.choices[0].message.content
