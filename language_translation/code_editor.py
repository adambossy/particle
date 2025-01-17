from pathlib import Path
from typing import Dict, List

from .conversation import Conversation
from .file_manager import FileManager

# Define the function schema for insert_code
INSERT_CODE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "insert_code",
        "description": "LLM function for inserting code into a source file",
        "parameters": {
            "type": "object",
            "properties": {
                "new_source": {
                    "type": "string",
                    "description": "The new source file with the code inserted",
                },
                "error": {
                    "type": "string",
                    "description": "Error message if insertion failed",
                    "default": "",
                },
            },
            "required": ["new_source"],
        },
    },
}


class CodeEditor(Conversation):
    """Applies edits to files in a repo using LLMs."""

    def __init__(self, file_manager: FileManager):
        super().__init__("anthropic/claude-3-sonnet-20240229")
        self.system_prompt = "You are an expert Go programmer."
        self.file_manager = file_manager

    def apply_edits(self, rel_fname_to_code: Dict[str, str]) -> str:
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

                def validate_insertion(response: dict) -> bool:
                    return (
                        not response.get("error")
                        and response.get("new_source") is not None
                        and "package" in response["new_source"]
                    )

                response = self.validated_completion(
                    prompt,
                    validate_insertion,
                    tools=[INSERT_CODE_FUNCTION],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "insert_code"},
                    },
                )

                print(f"\nResponse:\n{response}")
                new_source = response["new_source"]

                print(f"\nNew source:")
                print(new_source)
                print(f"--------------------------------")

            self.file_manager.rewrite_file(file_path, new_source)
            results.append(f"Processed {rel_fname}")

        return "\n".join(results)
