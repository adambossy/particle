import re
from pathlib import Path


class FileMap:
    """Specific for Python to Go translation"""

    def __init__(self, project_path: Path):
        self.py_project_path: Path = project_path
        self.go_project_path: Path = project_path.with_name(project_path.name + "_go")
        self.file_map = {}

    def setup_project(self):
        self.go_project_path.mkdir(parents=True, exist_ok=True)

    def _make_go_file_path(self, file_path: Path) -> Path:
        relative_file_path = file_path.relative_to(self.py_project_path)
        if relative_file_path.name.startswith("test_"):
            # Remove "test_" prefix if it exists and append "_test" before the extension
            new_filename = re.sub(r"^test_", "", relative_file_path.stem) + "_test"
            go_filename = relative_file_path.with_name(new_filename).with_suffix(".go")
        else:
            go_filename = relative_file_path.with_suffix(".go")
        full_path = self.go_project_path / go_filename
        return full_path

    # TODO (adam) This could be renamed "add_files"
    def setup_files(self, py_files: list[str]):
        for file in py_files:
            # For each file, we'll create a new file in the project_path
            # with the same name but with a .go extension
            py_file_path: Path = Path(file)
            relative_py_file_path: Path = py_file_path.relative_to(self.py_project_path)
            go_file_path: Path = self._make_go_file_path(py_file_path)

            self.file_map[relative_py_file_path] = go_file_path
            print(f"Creating {go_file_path} for {relative_py_file_path}")

            go_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not go_file_path.exists():
                with open(go_file_path, "w") as new_file:
                    new_file.write("")

        # TODO (adam) Init git repo, setup go.mod, etc.

    def get_translated_file_path(self, py_file_path: str) -> Path:
        cleaned_path = Path(py_file_path).relative_to(self.py_project_path)
        return self.file_map[cleaned_path]
