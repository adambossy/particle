import os

from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langsmith import traceable, wrappers

load_dotenv()


class Conversation:
    def __init__(self, llm: BaseChatModel):
        """
        Initialize Conversation with any LangChain-compatible LLM.

        Args:
            llm: A LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
        """
        # Wrap the LLM for tracing based on its type
        self.llm = llm
        # if isinstance(llm, ChatOpenAI):
        # self.llm = wrappers.wrap_openai(llm)
        # elif isinstance(llm, ChatAnthropic):
        #     self.llm = llm  # wrappers.wrap_anthropic(llm)
        # else:
        #     self.llm = llm  # Fallback for other LLM types

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

DEFAULT_SOURCE_AND_FILES = {
    "src/process_data.py": [
        """def process_data(data):
    '''Process the input data'''
    result = []
    for item in data:
        result.append(item * 2)
    return result

# test_process_data.py
import pytest
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
