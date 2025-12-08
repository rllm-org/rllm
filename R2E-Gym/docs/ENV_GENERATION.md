# Repository Analysis Guide

This guide documents the general process for analyzing Python repositories, setting up test environments, and constructing Docker containers for testing. We'll use examples from various repositories to illustrate the process.

## Prerequisites
- Python environment with `uv` installed
- Git for version control
- Docker for container management

## Process Overview
The repository analysis process involves:
1. Setting up the repository and test directories
2. Configuring the analysis system
3. Collecting and parsing commit history
4. Analyzing commits for test-related changes
5. Validating test environments using Docker

## Required Configuration
Before analyzing a new repository, you need to modify two configuration files:

### 1. Update Constants (r2egym/repo_analysis/constants.py)
Add your repository name to the list of supported repositories:
```python
repo_str_names = [
    "sympy",
    "pandas",
    # ... other repositories ...
    "your_repo_name",  # Add your repository here
]
```
This automatically sets up necessary directory structures and paths.

### 2. Update Repository Enum (r2egym/repo_analysis/repo_analysis_args.py)
Add your repository to the RepoName enum and configure test command:
```python
class RepoName(str, Enum):
    sympy = "sympy"
    pandas = "pandas"
    # ... other repositories ...
    your_repo_name = "your_repo_name"  # Add your repository here

class RepoAnalysisArgs(BaseModel):
    # ... other code ...
    @property
    def tests_cmd(self):
        # ... other conditions ...
        if self.repo_name == RepoName.your_repo_name:
            return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests"
```

## Detailed Steps

### 1. Repository Setup
First, add the target repository as a git submodule and create the necessary data directories:
```bash
# Create data directories for the target repository
mkdir -p commit_data/{repo_name} test_data/{repo_name}

# Add the repository as a submodule
git submodule add https://github.com/{org}/{repo_name}.git {repo_name}
```

For example, with the bokeh repository:
```bash
mkdir -p commit_data/bokeh test_data/bokeh
git submodule add https://github.com/bokeh/bokeh.git bokeh
```

### 2. Commit Collection and Analysis
The process involves two main steps:

#### 2.1 Collecting Commits
Process and store the repository's commit history:
```bash
uv run python r2egym/repo_analysis/store_repo_commits.py \
    --repo_name {repo_name} \
    --n_cpus 60
```

This step:
- Processes all commits in the repository
- Parses diffs and commit messages
- Stores commit data for analysis
- Uses parallel processing for efficiency

#### 2.2 Analyzing Testable Commits
Filter and analyze commits to identify test-related changes:
```bash
uv run python r2egym/repo_analysis/analyze_testable_commits.py \
    --repo_name {repo_name} \
    --use_local_commit_data \
    --n_cpus 50 \
    --N 5000 \
    --keep_only_bug_edit_commits \
    --keep_only_testmatch_commits \
    --keep_only_test_entity_edit_commits
```

The analysis applies several filters:
- Commit size (files/lines changed)
- File types (Python files)
- Commit purpose (bug fixes)
- Test-related changes
- Test entity modifications

### 3. Environment Validation
Test the installation process and validate the test environment:
```bash
uv run python r2egym/repo_analysis/repo_testextract.py \
    --repo_name {repo_name} \
    --use_local_commit_data \
    --n_cpus 50 \
    --N 500 \
    --keep_only_bug_edit_commits \
    --keep_only_testmatch_commits \
    --keep_only_test_entity_edit_commits \
    --model_name o1-mini \
    --max_tokens 12000
```

This step:
- Creates isolated Docker environments
- Validates installation scripts
- Tests commit changes
- Verifies test execution

## Performance Optimization
- Use `--N 500` for initial testing and validation
- Adjust `--n_cpus` based on available system resources
- Enable Docker cleanup to manage disk space
- Store results in organized directories:
  - Commit data: `commit_data/{repo_name}/`
  - Test results: `test_data/{repo_name}/`
