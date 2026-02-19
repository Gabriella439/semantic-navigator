import re
import os
import time

from contextlib import contextmanager

from semantic_navigator.models import Tree


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


@contextmanager
def timed(label: str, timings: dict[str, float] | None = None):
    """Context manager that prints elapsed time for a phase."""
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    print(f"{label} ({_fmt_time(elapsed)})")
    if timings is not None:
        timings[label] = elapsed


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain code fences or surrounding text."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


def repair_json(text: str) -> str:
    """Fix invalid escape sequences in JSON strings."""
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    text = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
    return text


def _sanitize_path(path: str) -> str:
    """Turn an absolute path into a safe directory name."""
    resolved = os.path.realpath(path)
    return resolved.replace("\\", "--").replace("/", "--").replace(":", "")


def case_insensitive_glob(pattern: str) -> str:
    return ''.join(
        f'[{c.upper()}{c.lower()}]' if c.isalpha() else c
        for c in pattern
    )


def to_pattern(files: list[str]) -> str:
    prefix = os.path.commonprefix(files)
    suffix = os.path.commonprefix([ file[len(prefix):][::-1] for file in files ])[::-1]

    middle_parts = [file[len(prefix):-len(suffix)] if suffix else file[len(prefix):] for file in files]
    star = "*" if any(middle_parts) else ""

    if not prefix and not suffix:
        return ""
    if not suffix:
        return f"{prefix}{star}: "
    if not prefix:
        return f"{star}{suffix}: "
    return f"{prefix}{star}{suffix}: "


def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]
