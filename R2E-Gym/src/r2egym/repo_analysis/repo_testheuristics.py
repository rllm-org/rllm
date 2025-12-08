import shutil
from pathlib import Path


def handle_pyramid_import(repo_path: Path):
    # check if "tests/" or "pyramid/tests" exists
    # for the test folder, loop on the test files, check if file contains sys.modules
    # if so prepend "import tests" or "import pyramid.tests" to the file

    r2e_tests_path = repo_path / "r2e_tests"

    tests_path = repo_path / "tests"
    pyramid_tests_path = repo_path / "pyramid/tests"

    final_tests_path = None

    if tests_path.exists():
        final_tests_path = tests_path

    if pyramid_tests_path.exists():
        final_tests_path = pyramid_tests_path

    for test_file in final_tests_path.iterdir():
        if not test_file.is_file():
            continue
        with open(test_file, "r") as f:
            content = f.read()
        if "sys.modules" in content:
            with open(test_file, "w") as f:
                f.write("import pyramid.tests\n" + content)

    # copy folders
    folder_names = ["fixtures", "test_config", "test_scripts"]

    for folder_name in folder_names:
        fixtures_path = final_tests_path / folder_name
        r2e_fixtures_path = r2e_tests_path / folder_name

        if fixtures_path.exists():
            shutil.copytree(fixtures_path, r2e_fixtures_path)


def handle_aiohttp_makefile(repo_path: Path):
    makefile_path = repo_path / "Makefile"

    ## replace `python -m pip install` and `pip install` with `uv pip install`

    with open(makefile_path, "r") as f:
        content = f.read()

    content = content.replace("python -m pip install", "pip install")
    content = content.replace("pip install", "uv pip install")

    with open(makefile_path, "w") as f:
        f.write(content)

    return


def repo_heuristics(repo_name: str, repo_path: Path):
    if repo_name == "pyramid":
        handle_pyramid_import(repo_path)
    if repo_name == "aiohttp":
        handle_aiohttp_makefile(repo_path)
    else:
        pass
