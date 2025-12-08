import re
import sys


def parse_pytest_output(output):
    """
    Parse pytest output and map test names to their complete error messages.

    :param output: Full pytest output as a string
    :return: Dictionary mapping test names to their error messages
    """
    test_blocks = {}

    # output = output.split("_ " * 20)[0]
    output = output.split("====== warnings summary =========")[0]
    output = (
        output.split("============================= PASSES ")[0]
        .strip()
        .strip("=")
        .strip()
    )

    # Split output into sections by 'ERRORS' and 'FAILURES', considering lines with '='
    pattern = r"\n=+\s+(ERRORS|FAILURES)\s+=+\n"
    split_sections = re.split(pattern, output)
    # split_sections contains [text_before, section_name1, section_content1, section_name2, section_content2, ...]

    for i in range(1, len(split_sections), 2):
        section_name = split_sections[i]
        section_content = split_sections[i + 1]
        # print(section_content)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Split the section_content into individual test error/failure outputs
        # They are separated by lines beginning with '_'
        test_sections = re.split(r"\n__+\s+", section_content)
        for test_section in test_sections:
            if not test_section.strip():
                continue
            # The first line is the test identifier
            lines = test_section.strip().splitlines()
            if not lines:
                continue
            test_line = lines[0].strip()
            test_line = test_line.lstrip("_").strip()
            test_name = ""
            if test_line.startswith("ERROR at setup of "):
                test_name = test_line[len("ERROR at setup of ") :].split()[0]
            else:
                test_name = test_line.split()[0]

            # NOTE: for parameterized tests we use
            # the function name without the parameter
            test_name = test_name.split("[")[0]

            # Collect the whole section (test_section)
            # Prepend the separator line to maintain the original format
            test_blocks[test_name] = ("\n".join(test_section.splitlines()[1:])).strip()
    return test_blocks
