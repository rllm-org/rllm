#!/root/.venv/bin/python
"""
Description: A simple finish tool with a "submit" command.

Notes about the `submit` command:
* When invoked with `--result`, the provided string is used for submitting required task results (e.g., localization files).
* If no `--result` is provided, it defaults to an empty string.

**Parameters:**
  1. **command** (`string`, required): The command to run. Currently allowed option is: `submit`.
     - Allowed value: [`submit`]
  2. **result** (`string`, optional): The result text to submit. Defaults to an empty string.
"""

import argparse
import sys


def submit(result: str = ""):
    """
    Submits a final result, printing a message that includes the result.
    """
    print("<<<Finished>>>")
    # if result:
    #     print(f"Final result submitted: {result}")
    # else:
    #     print("No result provided.")
    # You can add more logic here as needed


def main():
    parser = argparse.ArgumentParser(
        description="submit tool: run the `submit` command with an optional `--result` argument."
    )
    parser.add_argument("command", help="Subcommand to run (currently only `submit`).")
    parser.add_argument(
        "--result", help="The result text to submit (optional).", default=""
    )

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.result)
    else:
        print(f"Unknown command '{args.command}'. Only `submit` is supported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
