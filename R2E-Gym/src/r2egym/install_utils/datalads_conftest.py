import pytest
import tempfile
import os


@pytest.fixture
def path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def setup_git_config():
    with tempfile.TemporaryDirectory() as home:
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = home
        os.system('git config --global user.name "DataLad Tester"')
        os.system('git config --global user.email "test@example.com"')
        yield
        if old_home:
            os.environ["HOME"] = old_home
