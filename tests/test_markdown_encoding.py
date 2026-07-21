import unicodedata
from pathlib import Path

EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
)
ALLOWED_CONTROL_CHARACTERS = frozenset({"\t", "\n", "\r"})


def _versioned_markdown_files() -> list[Path]:
    return sorted(
        path
        for path in Path.cwd().rglob("*.md")
        if not EXCLUDED_DIRECTORIES.intersection(path.parts)
    )


def test_markdown_files_are_valid_utf8_plain_text() -> None:
    markdown_files = _versioned_markdown_files()
    assert markdown_files

    failures: list[str] = []
    for path in markdown_files:
        raw = path.read_bytes()
        if b"\x00" in raw:
            failures.append(f"{path}: contains a NUL byte")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            failures.append(f"{path}: invalid UTF-8 ({error})")
            continue

        forbidden_controls = sorted(
            {
                f"U+{ord(character):04X}"
                for character in text
                if character not in ALLOWED_CONTROL_CHARACTERS
                and unicodedata.category(character) == "Cc"
            }
        )
        if forbidden_controls:
            failures.append(
                f"{path}: contains forbidden control characters "
                f"{', '.join(forbidden_controls)}"
            )

    assert not failures, "\n".join(failures)
