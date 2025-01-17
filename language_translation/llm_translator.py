from pathlib import Path
from typing import Dict, List

import click
from dotenv import load_dotenv

from .conversation import Conversation
from .file_manager import FileManager

load_dotenv()

# Define the function schema for translate_code
TRANSLATE_CODE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "translate_code",
        "description": "Response from Python to Go translation",
        "parameters": {
            "type": "object",
            "properties": {
                "translated_source": {
                    "type": "string",
                    "description": "The translated Go code",
                },
                "error": {
                    "type": "string",
                    "description": "Error message if translation failed",
                    "default": "",
                },
            },
            "required": ["translated_source"],
        },
    },
}


class LLMTranslator(Conversation):
    def __init__(self, file_manager: FileManager):
        """
        Initialize LLMTranslator with LiteLLM model.
        """
        super().__init__()
        self.file_map = file_manager

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
        # Define the conversation with example
        self.prompt = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": """Translate this Python code and its tests to Go. 
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

def test_filter_and_transform():
    # Test with mixed positive and negative numbers
    assert filter_and_transform([-1, 0, 1, -2, 2]) == ["num: 0", "num: 1", "num: 2"]
    
    # Test with empty list
    assert filter_and_transform([]) == []
    
    # Test with all negative numbers
    assert filter_and_transform([-1, -2, -3]) == []""",
            },
            {
                "role": "assistant",
                "content": """// go_project/src/implementation.go
package main

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
package main

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
}""",
            },
        ]

    def translate(self, code_snippets_by_file: Dict[str, List[str]]) -> str:
        # Compose the prompt by appending each source code with its filename
        composed_prompt = """Translate this Python code and its tests to Go.
For each code snippet, prepend the translation with a comment containing the full relative file path
in the new repo that it belongs to, followed by the translated code. Use the file mappings that are
provided to map the file path to the new repo.

The package name should be the same as the enclosing directory of the file.

File Mappings:\n"""
        for py_filename in code_snippets_by_file.keys():
            py_file_path = Path(py_filename)
            go_file_path = self.file_map.get_target_file_path(py_file_path)
            composed_prompt += (
                f"{py_file_path.as_posix()} -> {go_file_path.as_posix()}\n"
            )

        composed_prompt += "\n"
        for filename, code_snippets in code_snippets_by_file.items():
            composed_prompt += f"# {filename}\n"
            for code_snippet in code_snippets:
                composed_prompt += f"{code_snippet}\n\n"
            composed_prompt += "\n"

        print("\nComposed prompt:")
        print(composed_prompt)

        def validate_translation(response: dict) -> bool:
            return (
                not response.get("error")
                and response.get("translated_source") is not None
                and "package" in response["translated_source"]
                and "func" in response["translated_source"]
            )

        # Create messages list with example conversation and new prompt
        messages = self.prompt + [{"role": "user", "content": composed_prompt}]

        response = self.validated_completion(
            messages,
            validate_translation,
            tools=[TRANSLATE_CODE_FUNCTION],
            tool_choice={
                "type": "function",
                "function": {"name": "translate_code"},
            },
        )

        return response["translated_source"]


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
