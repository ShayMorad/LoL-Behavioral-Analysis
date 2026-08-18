import os
from pathlib import Path


def print_tree(directory, ignore_dirs=None, prefix=""):
    # Add any other folders you want to skip to this list
    if ignore_dirs is None:
        ignore_dirs = {'.venv', '.idea', '.git', '__pycache__', 'dataset1', 'Games of League of Legends'}

    path = Path(directory)

    # Get all items, ignoring the ones in our list
    items = sorted([p for p in path.iterdir() if p.name not in ignore_dirs])

    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, ignore_dirs, prefix + extension)


print("project/")
print_tree(".")