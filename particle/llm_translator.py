import json
import os
import pprint
from pathlib import Path

import click
import instructor
import litellm
from dotenv import load_dotenv

from particle.utils import (
    get_assistant_message_from_tool_call,
    get_user_message_from_tool_call,
)

from .file_manager import FileManager

load_dotenv()

os.environ["LITELLM_LOG"] = "DEBUG"

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# litellm.set_verbose = True


def translate_code(translated_code: str, error: str) -> str:
    """
    Translate the code and return the updated translated source.
    """
    pass


# Use this for gpt-4o
gpt_4o_translate_code_tool = {
    "type": "function",
    "function": litellm.utils.function_to_dict(translate_code),
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
    "anthropic/claude-3-5-sonnet-20241022": claude_translate_code_tool,
}


class LLMTranslator:
    def __init__(self, model: str, file_manager: FileManager):
        super().__init__()

        self.model = model
        self.file_manager = file_manager

        self.client = instructor.from_litellm(litellm.acompletion)

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

import (
    "reflect"
    "testing"
)

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

    async def completion(self) -> None:
        completion = await litellm.acompletion(
            messages=self.messages,
            model=self.model,
            tools=[translate_code_tool_table[self.model]],
            tool_choice={"type": "function", "function": {"name": "translate_code"}},
            temperature=1.0,
        )

        self.last_completion = completion  # HACK?

        self.messages.append(
            get_assistant_message_from_tool_call(self.model, completion)
        )

        tool_call = completion.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        return arguments

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

        translate_code_response = await self.completion()
        return translate_code_response["translated_code"]

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

        response = await self.completion()
        return response["translated_code"]

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
    llm = MODELS[model]()
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
