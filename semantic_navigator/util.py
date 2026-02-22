import os

from semantic_navigator.models import Tree


def to_pattern(files: list[str]) -> str:
    prefix = os.path.commonprefix(files)
    suffix = os.path.commonprefix([ file[len(prefix):][::-1] for file in files ])[::-1]

    if suffix:
        if any([ file[len(prefix):-len(suffix)] for file in files ]):
            star = "*"
        else:
            star = ""
    else:
        if any([ file[len(prefix):] for file in files ]):
            star = "*"
        else:
            star = ""

    if prefix:
        if suffix:
            return f"{prefix}{star}{suffix}: "
        else:
            return f"{prefix}{star}: "
    else:
        if suffix:
            return f"{star}{suffix}: "
        else:
            return ""


def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]
