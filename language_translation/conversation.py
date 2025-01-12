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
        model="gpt-4o", temperature=1.0, api_key=os.getenv("OPENAI_API_KEY")
    ),
    "claude": lambda: ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=1.0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ),
}
