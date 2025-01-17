import os

from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
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
        self.llms = [llm, MODELS["gpt-4o"](), MODELS["gemini"]()]

    def primary_llm(self):
        return self.llms[0]

    def validated_completion(
        self, prompt: str, validator: callable, attempts_per_model: int = 1
    ) -> any:
        """
        Get a completion and validate it, falling back to other models if validation fails.

        Args:
            prompt: The input prompt
            validator: Function that takes completion result and returns bool
            max_attempts: Maximum number of retry attempts across all models

        Returns:
            The validated completion result

        Raises:
            Exception: If validation fails for all models after max attempts
        """
        for model in self.llms:
            for _ in range(attempts_per_model):
                print(f"Making completion call with {model.model}")
                result = self.completion(prompt, model)
                if validator(result):
                    return result

        raise Exception(
            f"Failed to get valid completion after {attempts_per_model} attempts"
        )

    @traceable
    def completion(self, prompt: str, llm: BaseChatModel | None = None) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: The input prompt requesting Python to Go translation

        Returns:
            The translated Go code
        """
        if not llm:
            llm = self.primary_llm
        chain = self.prompt | llm
        response = chain.invoke({"input": [HumanMessage(content=prompt)]})
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
    "gemini": lambda: ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=1.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ),
}
