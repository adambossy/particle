import re
from pathlib import Path


class FileManager:
    """Specific for Python to Go translation"""

    def __init__(self, project_path: Path):
        self.py_project_path: Path = project_path
        self.go_project_path: Path = project_path.with_name(project_path.name + "_go")
        self.file_map: dict[Path, Path] = {}

    def setup_project(self):
        self.go_project_path.mkdir(parents=True, exist_ok=True)

    def _make_go_file_path(self, rel_py_file_path: Path) -> Path:
        if rel_py_file_path.name.startswith("test_"):
            # Remove "test_" prefix if it exists and append "_test" before the extension
            new_filename = re.sub(r"^test_", "", rel_py_file_path.stem) + "_test"
            return rel_py_file_path.with_name(new_filename).with_suffix(".go")
        else:
            return rel_py_file_path.with_suffix(".go")

    # TODO (adam) This could be renamed "add_files"
    def setup_files(self, py_files: list[Path]):
        for rel_py_file_path in py_files:
            # For each file, we'll create a new file in the project_path
            # with the same name but with a .go extension
            rel_go_file_path: Path = self._make_go_file_path(rel_py_file_path)

            self.file_map[rel_py_file_path] = rel_go_file_path
            print(f"Creating {rel_go_file_path} for {rel_py_file_path}")

            full_go_file_path = self.go_project_path / rel_go_file_path
            full_go_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_go_file_path.exists():
                with open(full_go_file_path, "w") as new_file:
                    new_file.write("")

        # TODO (adam) Init git repo, setup go.mod, etc.

    def get_translated_file_path(self, py_file_path: str) -> Path:
        cleaned_path = Path(py_file_path).relative_to(self.py_project_path)
        return self.file_map[cleaned_path]

    def get_abs_py_file_path(self, py_file_path: Path) -> Path:
        return py_file_path.absolute()

    def get_abs_go_file_path(self, go_file_path: Path) -> Path:
        return go_file_path.absolute()

    def get_rel_py_file_path(self, py_file_path: Path) -> Path:
        return py_file_path.relative_to(self.py_project_path)

    def get_rel_go_file_path(self, go_file_path: Path) -> Path:
        return go_file_path.relative_to(self.go_project_path)

    def get_source_repo_file_paths(self) -> list[Path]:
        return list(self.file_map.keys())

    def get_target_repo_file_paths(self) -> list[Path]:
        return list(self.file_map.values())

    def get_target_file_path(self, py_file_path: Path) -> Path:
        return self.file_map[py_file_path]

    def insert_code(self, file_path: Path, code: str, line_number: int):
        with open(file_path, "r") as file:
            lines = file.readlines()

        lines.insert(line_number, code + "\n")

        with open(file_path, "w") as file:
            file.writelines(lines)

    def rewrite_file(self, file_path: Path, new_source: str):
        with open(file_path, "w") as file:
            file.write(new_source)

    def get_target_repo_path(self) -> Path:
        """Return the path to the target Go project."""
        return self.go_project_path
