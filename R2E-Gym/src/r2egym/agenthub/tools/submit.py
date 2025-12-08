#!/root/.venv/bin/python
"""
Description: A simple submit tool to finish tasks.

This tool signals completion of a task or submission of results.
No parameters required - simply call to indicate task completion.
"""

import argparse
import sys


def submit():
    """
    Submits completion signal.
    """
    print("<<<Finished>>>")


def main():
    parser = argparse.ArgumentParser(
        description="Submit tool: Signal task completion."
    )
    # No arguments needed
    args = parser.parse_args()
    
    submit()


if __name__ == "__main__":
    main()
