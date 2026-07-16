"""Build a single self-contained attack.py for Kaggle submission.

Dev-time submission code is split across sibling modules for testability. Kaggle
runs a lone attack.py with no sibling package, so we inline those modules into one
file, stripping intra-package imports.
"""

import re
from pathlib import Path

_SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission"
_MODULE_ORDER = ("templates", "engine", "selector", "attack")

_INTRA_IMPORT = re.compile(r"^\s*from jed_attack[.\w]* import .*$", re.MULTILINE)
_INTRA_IMPORT_MOD = re.compile(r"^\s*import jed_attack.*$", re.MULTILINE)
_FUTURE = re.compile(r"^\s*from __future__ import annotations\s*$", re.MULTILINE)
_MODULE_DOCSTRING = re.compile(r'\A\s*""".*?"""', re.DOTALL)


def _clean_module(name: str) -> str:
    """Return a module's source with intra-package imports and headers stripped.

    Args:
        name: Submission module name (without package prefix).

    Returns:
        Cleaned source ready for concatenation.
    """
    source = (_SUBMISSION_DIR / f"{name}.py").read_text(encoding="utf-8")
    source = _MODULE_DOCSTRING.sub("", source, count=1)
    source = _INTRA_IMPORT.sub("", source)
    source = _INTRA_IMPORT_MOD.sub("", source)
    source = _FUTURE.sub("", source)
    return source.strip() + "\n"


def build_attack_source() -> str:
    """Assemble the inlined, self-contained submission source.

    Returns:
        A single-module Python source string with no ``jed_attack`` imports.
    """
    header = (
        '"""JED red-team submission (auto-generated; do not edit).\n\n'
        "Built by jed_attack.scripts.build_submission from the submission package.\n"
        '"""\n\n'
    )
    bodies = "\n\n".join(_clean_module(name) for name in _MODULE_ORDER)
    return header + bodies + "\n"


def write_submission(out_dir: Path) -> Path:
    """Write attack.py and a Kaggle notebook cell to ``out_dir``.

    Args:
        out_dir: Output directory for the built files.

    Returns:
        The path to the written ``attack.py``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    attack_path = out_dir / "attack.py"
    attack_path.write_text(build_attack_source(), encoding="utf-8")
    notebook_cell = (
        "# Kaggle submission cell: writes attack.py to /kaggle/working/\n"
        "from pathlib import Path\n\n"
        "ATTACK_SOURCE = r'''\n" + build_attack_source() + "'''\n\n"
        "Path('/kaggle/working/attack.py').write_text(ATTACK_SOURCE)\n"
        "print('attack.py written')\n"
    )
    (out_dir / "submission_notebook.py").write_text(notebook_cell, encoding="utf-8")
    return attack_path


def main() -> None:
    """Write the submission to ``dist/``."""
    path = write_submission(Path("dist"))
    print(f"wrote {path}")  # noqa: T201


if __name__ == "__main__":
    main()
