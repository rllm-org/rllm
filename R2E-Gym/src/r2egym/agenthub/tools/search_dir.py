#!/root/.venv/bin/python

# @yaml
# signature: search_dir <search_term> [<dir>]
# docstring: Searches for search_term in all files in dir. If dir is not provided, searches in the current directory.
# arguments:
#   search_term:
#     type: string
#     description: The term to search for.
#     required: true
#   dir:
#     type: string
#     description: The directory to search in (if not provided, searches in the current directory).
#     required: false

import sys
import os


def main():
    # Access command-line arguments directly
    args = sys.argv[1:]  # Exclude the script name
    num_args = len(args)

    if num_args == 1:
        search_term = args[0]
        directory = "."
    elif num_args == 2:
        search_term = args[0]
        directory = args[1]
        if not os.path.isdir(directory):
            print(f"Directory {directory} not found")
            sys.exit(1)
    else:
        print("Usage: search_dir <search_term> [<dir>]")
        sys.exit(1)

    directory = os.path.realpath(directory)
    matches = {}
    num_files_matched = 0

    for root, dirs, files in os.walk(directory):
        # Exclude hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.startswith("."):
                continue  # Skip hidden files
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    file_matches = 0
                    for line_num, line in enumerate(f, 1):
                        if search_term in line:
                            file_matches += 1
                    if file_matches > 0:
                        matches[filepath] = file_matches
                        num_files_matched += 1
            except (UnicodeDecodeError, PermissionError):
                continue  # Skip files that can't be read

    if not matches:
        print(f'No matches found for "{search_term}" in {directory}')
        sys.exit(0)

    num_matches = sum(matches.values())

    if num_files_matched > 100:
        print(
            f'More than {num_files_matched} files matched for "{search_term}" in {directory}. Please narrow your search.'
        )
        sys.exit(0)

    print(f'Found {num_matches} matches for "{search_term}" in {directory}:')

    for filepath, count in matches.items():
        # Replace leading path with './' for consistency
        relative_path = os.path.relpath(filepath, start=os.getcwd())
        if not relative_path.startswith("./"):
            relative_path = "./" + relative_path
        print(f"{relative_path} ({count} matches)")

    print(f'End of matches for "{search_term}" in {directory}')


if __name__ == "__main__":
    main()
