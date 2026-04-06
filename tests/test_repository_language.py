import ast
import re
import unittest
from pathlib import Path


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
PIPELINE_DIR = Path("pipeline")


def _iter_docstrings(module_path: Path) -> list[tuple[str, str]]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    docstrings: list[tuple[str, str]] = []

    module_docstring = ast.get_docstring(module, clean=False)
    if module_docstring:
        docstrings.append((f"{module_path}:module", module_docstring))

    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring:
            docstrings.append((f"{module_path}:{node.name}", docstring))

    return docstrings


class RepositoryLanguageTests(unittest.TestCase):
    def test_pipeline_docstrings_are_written_in_russian(self):
        missing_russian: list[str] = []

        for module_path in sorted(PIPELINE_DIR.glob("*.py")):
            for location, docstring in _iter_docstrings(module_path):
                if not CYRILLIC_RE.search(docstring):
                    missing_russian.append(location)

        self.assertFalse(
            missing_russian,
            "These docstrings must contain Russian text: " + ", ".join(missing_russian),
        )
