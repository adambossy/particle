from .file_map import FileMap


class CodeEditor:
    """Applies edits to files in a repo.

    The edits are derived from LLM responses."""

    def __init__(self, file_map: FileMap):
        self.file_map = file_map

    def appy_edits(self, edits_by_file: dict[str, list[tuple[str, str]]]) -> str:
        pass
