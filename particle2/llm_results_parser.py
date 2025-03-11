import re

DEFAULT_FENCE = ("`" * 3, "`" * 3)

HEAD = r"^<{5,9} SEARCH\s*$"
DIVIDER = r"^={5,9}\s*$"
UPDATED = r"^>{5,9} REPLACE\s*$"

HEAD_ERR = "<<<<<<< SEARCH"
DIVIDER_ERR = "======="
UPDATED_ERR = ">>>>>>> REPLACE"

missing_filename_err = (
    "Bad/missing filename. The filename must be alone on the line before the opening fence"
    " {fence[0]}"
)


class LLMResultsParser:
    """Extracts code changes from LLM responses that can be applied to files."""

    def __init__(self):
        pass

    def parse_translations(self, translated_code: str) -> dict[str, str]:
        # TODO (adam) May want to use a sentinel that can't appear in code to avoid false positives
        file_pattern = re.compile(r"//\s*(.+\.go)")
        file_to_code_chunks = {}
        current_file = None

        for line in translated_code.splitlines():
            match = file_pattern.match(line)
            if match:
                current_file = match.group(1)
                file_to_code_chunks.setdefault(current_file, [])
            elif current_file:
                file_to_code_chunks[current_file].append(line)

        # Convert lists of lines into single code chunks
        for file in file_to_code_chunks:
            file_to_code_chunks[file] = "\n".join(file_to_code_chunks[file])

        return file_to_code_chunks
