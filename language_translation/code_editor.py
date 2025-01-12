import json
from pathlib import Path

from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable
from pydantic import BaseModel, Field

from language_translation.conversation import MODELS, Conversation
from language_translation.file_manager import FileManager


class insert_code(BaseModel):
    """LLM function for inserting code into a source file."""

    # new_code: str = Field(..., description="Go code to insert into the source file")
    # source_file: str = Field(
    #     ..., description="The file that we want to insert our new code into"
    # )
    new_source: str = Field(
        ..., description="The new source file with the code inserted"
    )
    error: str = Field("", description="Error message if insertion failed")


class CodeEditor(Conversation):
    """Applies edits to files in a repo.

    The edits are derived from LLM responses and also made using LLMs."""

    def __init__(self, file_manager: FileManager):
        llm = MODELS["claude"]()

        super().__init__(llm)  # Initialize the parent Conversation class

        # Bind the translation tool to the LLM
        self.llm = self.llm.bind_tools([insert_code], tool_choice="insert_code")

        self.system_prompt = """You are an expert Go programmer."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="input"),
            ]
        )
        self.chain = (
            self.prompt
            | self.llm
            # | PydanticToolsParser(
            #     tools=[insert_code],
            # )
        )

        self.file_manager = file_manager

    def apply_edits(self, rel_fname_to_code: dict[str, str]) -> str:
        """Apply edits to multiple files based on LLM guidance.

        Args:
            rel_fname_to_code: Dict mapping relative filenames to new code to insert

        Returns:
            Summary of changes made
        """
        results = []

        for rel_fname, new_code in rel_fname_to_code.items():
            file_path = self.file_manager.go_project_path / rel_fname

            # Read existing file contents
            with open(file_path) as f:
                source_code = f.read()

            is_empty = source_code.strip() == ""
            if is_empty:
                new_source = new_code
            else:
                print(f"\n===============================")
                print(f"File path: {file_path.absolute()} {file_path.as_posix()}")
                print(f"Source code:\n{source_code}")
                print(f"\n===============================")

                # Ask LLM where to insert the code
                prompt = f"""I have some new code to insert into an existing Go file.
    Please insert this code based on Go best practices, readability, and logical flow.
    Minimize the changes to only what's in the new source code.

    If the existing file is not empty, retain the existing package name.

    New code to insert:
    ```go
    {new_code}
    ```

    Existing source file:
    {rel_fname}
    ```go
    {source_code}
    ```
    """

                print(f"\nPrompt:\n{prompt}")

                response = self.completion(prompt)
                print(f"\nResponse:\n{response}")

                # Check for errors in the response
                if response.get("error"):
                    raise Exception(f"Error in response: {response.get('error')}")

                if not response.get("new_source"):
                    raise Exception(f"No insertions found in response: {response}")

                # Sort insertions in reverse order by line_number
                new_source = response.get("new_source")

                print(f"\nNew source:")
                print(new_source)
                print(f"--------------------------------")

            self.file_manager.rewrite_file(file_path, new_source)

            results.append(f"Processed {rel_fname}")

        return "\n".join(results)
