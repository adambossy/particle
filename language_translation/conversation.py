import os

import click
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith import traceable, wrappers
from pydantic import BaseModel, Field

load_dotenv()


class translate_code(BaseModel):
    """Response from Python to Go translation."""

    translated_source: str = Field(..., description="The translated Go code")
    error: str = Field("", description="Error message if translation failed")


class Conversation:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Conversation with any LangChain-compatible LLM.

        Args:
            llm: A LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
        """
        # Wrap the LLM for tracing based on its type
        if isinstance(llm, ChatOpenAI):
            self.llm = wrappers.wrap_openai(llm)
        elif isinstance(llm, ChatAnthropic):
            self.llm = llm  # wrappers.wrap_anthropic(llm)
        else:
            self.llm = llm  # Fallback for other LLM types

        # Bind the translation tool to the LLM
        self.llm = self.llm.bind_tools([translate_code], tool_choice="translate_code")

        # Define the system prompt
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

Remember: The translated code must pass all provided tests after conversion."""

        # Create the conversation chain with example
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content="""Translate this Python code and its test to Go:

# implementation.py
def filter_and_transform(items):
    '''
    Filter out negative numbers and transform the remaining ones.
    Returns a list of strings in the format "num: <value>"
    '''
    return [f"num: {x}" for x in items if x >= 0]

# test_implementation.py
import pytest

def test_filter_and_transform():
    # Test with mixed positive and negative numbers
    assert filter_and_transform([-1, 0, 1, -2, 2]) == ["num: 0", "num: 1", "num: 2"]
    
    # Test with empty list
    assert filter_and_transform([]) == []
    
    # Test with all negative numbers
    assert filter_and_transform([-1, -2, -3]) == []"""
                ),
                AIMessage(
                    content="""package main

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

// implementation_test.go
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
}"""
                ),
                MessagesPlaceholder(variable_name="input"),
            ]
        )

        # Create chain without itemgetter
        self.chain = self.prompt | self.llm

    @traceable
    def completion(self, prompt: str) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: The input prompt requesting Python to Go translation

        Returns:
            The translated Go code
        """
        response = self.chain.invoke({"input": [HumanMessage(content=prompt)]})
        return response.tool_calls[0]["args"]


MODELS = {
    "gpt-4o": lambda: ChatOpenAI(
        model="gpt-4o", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
    ),
    "claude": lambda: ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ),
}

DEFAULT_PROMPT = """Translate this Python code to Go:
def process_data(data):
    '''Process the input data'''
    result = []
    for item in data:
        result.append(item * 2)
    return result

# test_process_data.py
import pytest

def test_process_data():
    # Test with a list of positive numbers
    assert process_data([1, 2, 3]) == [2, 4, 6]
    
    # Test with an empty list
    assert process_data([]) == []
    
    # Test with a list containing zero
    assert process_data([0, 1, 2]) == [0, 2, 4]
"""


@click.command()
@click.option(
    "--model",
    type=click.Choice(["gpt-4o", "claude"]),
    default="claude",
    help="The LLM model to use",
)
@click.option("--prompt", default=DEFAULT_PROMPT, help="The prompt to send to the LLM")
def cli(model: str, prompt: str) -> None:
    """CLI tool for translating Python code to Go."""
    llm = MODELS[model]()
    conversation = Conversation(llm)

    click.echo("\nInput Python function:")
    click.echo(
        prompt.split("Translate this Python code to Go:\n")[1]
    )  # Extract just the function part
    click.echo("\nTranslated Go code:")
    result = conversation.completion(prompt)

    click.echo("\nResult:")
    click.echo(result["translated_source"])


if __name__ == "__main__":
    cli()
