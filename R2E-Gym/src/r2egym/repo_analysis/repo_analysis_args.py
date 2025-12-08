from enum import Enum
from pydantic import BaseModel, Field

from r2e.llms import LLMArgs
from r2egym.repo_analysis.constants import *


class RepoName(str, Enum):
    sympy = "sympy"
    pandas = "pandas"
    numpy = "numpy"
    scrapy = "scrapy"
    tornado = "tornado"
    statsmodels = "statsmodels"
    pillow = "pillow"
    pyramid = "pyramid"
    datalad = "datalad"
    aiohttp = "aiohttp"
    mypy = "mypy"
    coveragepy = "coveragepy"
    orange3 = "orange3"
    bokeh = "bokeh"


class RepoAnalysisArgs(BaseModel):
    repo_name: RepoName
    n_cpus: int = Field(32, ge=1)
    use_local_commit_data: bool = Field(True)

    @property
    def repo_dir(self):
        return globals()[self.repo_name.upper() + "_DIR"]

    @property
    def gcp_commit_data_dir(self):
        return globals()[self.repo_name.upper() + "_COMMIT_DATA_DIR"]

    @property
    def local_commit_data_dir(self):
        return globals()["LOCAL_" + self.repo_name.upper() + "_COMMIT_DATA_DIR"]

    @property
    def commit_data_dir(self):
        return (
            self.local_commit_data_dir
            if self.use_local_commit_data
            else self.gcp_commit_data_dir
        )

    @property
    def test_data_dir(self):
        return globals()[self.repo_name.upper() + "_TEST_DATA_DIR"]

    @property
    def parameterized_dockerfile(self):
        return f"src/r2e_edits/repo_analysis/base_dockerfiles/Dockerfile.{self.repo_name.value}"

    ## TODO: remove these old install configurations, now we use `install_utils/{repo_name}_install.sh`
    # @property
    # def envsetup_cmd(self):
    #     if self.repo_name == RepoName.sympy:
    #         return "uv venv --python=python3.8"
    #     if self.repo_name == RepoName.pandas:
    #         return "uv venv --python=python3.10"
    #     if self.repo_name == RepoName.statsmodels:
    #         return "source .venv/bin/activate"
    #     raise NotImplementedError("only sympy is supported for now")

    # @property
    # def preinstall_cmd(self):
    #     if self.repo_name == RepoName.sympy:
    #         return "source .venv/bin/activate && uv pip install numpy mpmath pytest ipython numexpr"
    #     if self.repo_name == RepoName.pandas:
    #         VERSIONEER_COMMAND = 'echo -e "[versioneer]\nVCS = git\nstyle = pep440\nversionfile_source = pandas/_version.py\nversionfile_build = pandas/_version.py\ntag_prefix =\nparentdir_prefix = pandas-" > setup.cfg && versioneer install'
    #         return f'source .venv/bin/activate && uv pip install --upgrade setuptools "numpy==0.17.*" "cython<=0.30" python-dateutil pytz pytest hypothesis versioneer && {VERSIONEER_COMMAND}'
    #     if self.repo_name == RepoName.statsmodels:
    #         return "uv pip install -r numpy scipy pandas patsy"
    #     raise NotImplementedError("only sympy is supported for now")

    # @property
    # def install_cmd(self):
    #     if self.repo_name == RepoName.sympy:
    #         return "source .venv/bin/activate && uv pip install -e ."
    #     if self.repo_name == RepoName.pandas:
    #         return "source .venv/bin/activate && rm -f pyproject.toml && CFLAGS='-O0 -Wno-error=array-bounds' uv run python setup.py build_ext --inplace -j 4"
    #     if self.repo_name == RepoName.statsmodels:
    #         return "source .venv/bin/activate && uv pip install -e ."
    #     raise NotImplementedError("only sympy is supported for now")

    @property
    def tests_cmd(self):
        if self.repo_name == RepoName.sympy:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.pandas:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.pillow:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.scrapy:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.pyramid:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.datalad:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.aiohttp:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.coveragepy:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.numpy:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.orange3:
            return "QT_QPA_PLATFORM=minimal PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' xvfb-run --auto-servernum .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        if self.repo_name == RepoName.bokeh:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
        raise NotImplementedError("only sympy is supported for now")


class RepoAnalysisLoadArgs(RepoAnalysisArgs):
    N: int | None = Field(None)
    load_verbose: bool = Field(True)
    load_run_parallel: bool = Field(True)

    # filters
    keep_only_small_commits: bool = Field(True)
    keep_only_python_commits: bool = Field(True)
    keep_only_non_docstring_commits: bool = Field(True)
    keep_only_bug_edit_commits: bool = Field(False)
    keep_only_testmatch_commits: bool = Field(False)
    keep_only_test_entity_edit_commits: bool = Field(False)

    ## repo specific
    ### pandas date
    keep_pandas_year_cutoff: bool = Field(False)
    ### mypy test file edit
    keep_only_mypy_test_edit: bool = Field(False)

    # heuristics

    ## long
    max_num_non_test_files: int = Field(5, ge=0)
    max_num_non_test_edited_lines: int = Field(200, ge=0)
    max_patch_length: int = Field(10000, ge=0)

    ## bugedit
    max_num_nontest_deleted_entities: int = Field(0, ge=0)
    max_num_nontest_added_entities: int = Field(1, ge=0)
    max_num_nontest_edited_entities: int = Field(4, ge=0)
    max_num_statement_entities: int = Field(6, ge=0)


class RepoAnalysisTestExtractArgs(RepoAnalysisLoadArgs, LLMArgs):
    clean_old_runs: bool = Field(False)

    chunk_size: int = Field(50, ge=1)
    start_chunk: int = Field(0, ge=0)

    rebuild_dockers: bool = Field(True)
    build_dockers: bool = Field(False)
    push_dockers: bool = Field(False)
    cleanup_dockers: bool = Field(False)

    n_cpus_docker: int = Field(10, ge=1)
