import instructor
import litellm
from pydantic import BaseModel

from language_translation.utils import get_assistant_message_from_tool_call

from .file_manager import FileManager

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


class InsertCodeResponse(BaseModel):
    new_source: str
    error: str = ""


def validate_insertion(response: InsertCodeResponse) -> bool:
    return (
        not response.error
        and response.new_source is not None
        and "package" in response.new_source
    )


class CodeEditor:
    """Applies edits to files in a repo using LLMs."""

    def __init__(self, model: str, file_manager: FileManager):
        super().__init__()

        self.model = model
        self.file_manager = file_manager

        self.client = instructor.from_litellm(litellm.completion)

        self.system_prompt = "You are an expert Go programmer."

        self.initialize_messages()

    def completion(self, node_name: str) -> None:
        trace_id = f"insert-{node_name}-{self.model}"

        print(f"\n--- PROMPT ---")
        print(self.messages[-1]["content"])
        print(f"--- END PROMPT ---")

        response, completion = self.client.chat.completions.create_with_completion(
            messages=self.messages,
            model=self.model,
            response_model=InsertCodeResponse,
            max_retries=2,
            # metadata={
            #     "trace_id": trace_id,
            # },
        )

        print(f"\n--- RESPONSE ---")
        if response.error:
            print(response.error)
        elif response.new_source:
            print(response.new_source)
        print(f"--- END RESPONSE ---")

        self.messages.append(
            get_assistant_message_from_tool_call(self.model, completion)
        )

        return response

    def compose_prompt(self, rel_fname: str, new_code: str, source_code: str) -> str:
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
        return prompt

    def insert_code(self, rel_fname_to_code: dict[str, str], node_name: str) -> str:
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
                prompt = self.compose_prompt(rel_fname, new_code, source_code)
                self.messages.append({"role": "user", "content": prompt})
                insert_code_response = self.completion(node_name)
                new_source = insert_code_response.new_source

            self.file_manager.rewrite_file(file_path, new_source)
            results.append(f"Inserted code into file: {rel_fname}")

        return "\n".join(results)

    def retry(self, last_test_output: str, node_name: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": last_test_output
                + """\n\nPlease try inserting the code again, providing an updated insertion that fixes these test failures.

Remember to insert this code based on Go best practices, readability, and logical flow.
Minimize the changes to only what's in the new source code.

If the existing file is not empty, retain the existing package name.
                """,
            }
        )

        print(f"\n--- RETRY PROMPT ---")
        print(self.messages[-1]["content"])
        print(f"--- END RETRY PROMPT ---")

        response = self.completion(node_name)

        print(f"\n--- RETRY RESPONSE ---")
        if response.error:
            print(response.error)
        elif response.new_source:
            print(response.new_source)
        print(f"--- END RETRY RESPONSE ---")

        return response.new_source

    def initialize_messages(self) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": self.one_shot_user_message},
            # {"role": "assistant", "content": self.one_shot_assistant_message},
        ]

    def clear_messages(self) -> None:
        self.initialize_messages()
