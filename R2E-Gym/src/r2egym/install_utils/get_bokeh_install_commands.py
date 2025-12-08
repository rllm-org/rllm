#!/usr/bin/env python3
import itertools
import os
import subprocess
import sys
import tempfile
import shutil
import concurrent.futures
import random

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------
BOKEH_REPO_URL = "https://github.com/bokeh/bokeh.git"

# Which bokeh versions/tags to build:
BOKEH_VERSIONS = [
    # Older versions
    "1.4.0",
    "2.0.0",
    "2.1.1",
    "2.2.3",
    "2.3.3",
    "2.4.3",
    # Newer versions
    "3.0.0",
    "3.0.3",
    "3.1.0", 
    "3.2.0",
    "3.2.2", 
    "3.3.0",
    "3.3.2",
    "3.4.0",
]

# For testing, just use a few versions
TEST_VERSIONS = [
    "2.0.0",  # Older version
    "2.4.3",  # Middle version
    "3.4.0",  # Latest version
]

# Dependency versions to try in combination - more options for older versions
PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"] # Include older Python versions
JINJA2_VERSIONS = [">=2.9", "==2.10.3"]  # Try different Jinja2 versions
NUMPY_VERSIONS = [">=1.12", ">=1.16", "==1.19.5"]  # Try different NumPy versions
PANDAS_VERSIONS = [">=0.25.3", ">=1.0", ">=1.2"]  # Try older Pandas
PILLOW_VERSIONS = [">=6.0.0", ">=7.1.0"]  # Try older Pillow
TORNADO_VERSIONS = [">=5.1", ">=6.0", ">=6.2"]  # Try older Tornado
CONTOURPY_VERSIONS = [">=1.0.0", ">=1.2"]  # For newer Bokeh versions

# For older Bokeh versions (pre 3.0), we may need to not use contourpy at all
# We'll handle this in the build function

# Full versions list (uncomment for production use)
"""
PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
JINJA2_VERSIONS = [">=2.9", "==2.11.3", "==3.0.0", "==3.1.2"]
NUMPY_VERSIONS = [">=1.16", ">=1.20", ">=1.22", ">=1.24", ">=1.26"]
PANDAS_VERSIONS = [">=1.2", ">=1.3", ">=1.5", ">=2.0", ">=2.1"]
PILLOW_VERSIONS = [">=7.1.0", ">=9.0.0", ">=10.0.0"]
TORNADO_VERSIONS = [">=6.2", "==6.2", "==6.3.2"]
CONTOURPY_VERSIONS = [">=1.2", "==1.2.0", "==1.2.1"]
"""

# Where to store overall build results:
BUILD_RESULTS_FILE = "bokeh_build_results.txt"

# ------------------------------------------------------------------------------
#  LOGGING UTILS
# ------------------------------------------------------------------------------


def log_and_run(cmd, logfile_path, *, shell=True, check=True, timeout=900):
    """
    Logs the command to console and also appends it to logfile_path, then runs it.
    """
    print(f"[RUN] {cmd}")
    with open(logfile_path, "a", encoding="utf-8") as f:
        f.write(f"{cmd}\n")
    try:
        return subprocess.run(
            cmd, shell=shell, check=check, timeout=timeout, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as e:
        with open(logfile_path, "a", encoding="utf-8") as f:
            f.write(f"COMMAND FAILED: {e}\n")
            if hasattr(e, 'stdout') and e.stdout:
                f.write(f"STDOUT:\n{e.stdout}\n")
            if hasattr(e, 'stderr') and e.stderr:
                f.write(f"STDERR:\n{e.stderr}\n")
        raise
    except subprocess.TimeoutExpired as e:
        with open(logfile_path, "a", encoding="utf-8") as f:
            f.write(f"COMMAND TIMED OUT: {e}\n")
        raise


# ------------------------------------------------------------------------------
#  BUILD LOGIC
# ------------------------------------------------------------------------------


def build_bokeh_with_versions(
    python_ver, jinja2_ver, numpy_ver, pandas_ver, tornado_ver, 
    pillow_ver, contourpy_ver, logfile_path
):
    """
    Attempt to build the local checkout of bokeh using a new .venv with
    specified versions of Python and dependencies.
    Returns True if the build succeeds, False otherwise.
    """
    venv_dir = os.path.join(os.getcwd(), ".venv")
    if os.path.exists(venv_dir):
        shutil.rmtree(venv_dir)

    print(f"Trying: py={python_ver}, jinja2={jinja2_ver}, numpy={numpy_ver}, pandas={pandas_ver}, tornado={tornado_ver}, pillow={pillow_ver}, contourpy={contourpy_ver}")
    
    try:
        # Create .venv with the given Python version
        cmd_create_venv = f"uv venv --python {python_ver}"
        log_and_run(cmd_create_venv, logfile_path)

        # We'll assume .venv is created in the current directory:
        activate_path = os.path.join(".venv", "bin", "activate")

        def run_in_venv(cmd, **kwargs):
            full_cmd = f"bash -c 'set -e; source \"{activate_path}\" && {cmd}'"
            return log_and_run(full_cmd, logfile_path, **kwargs)

        # Upgrade pip/wheel inside the venv
        run_in_venv("uv pip install --upgrade pip wheel setuptools")

        # For older Bokeh versions (< 3.0), don't install contourpy and narwhals
        # Check the git directory to determine the version (since we know we're in a git checkout)
        version_major = 3  # Default to new version
        
        # Try to detect from setup.py or pyproject.toml
        if os.path.exists('setup.py'):
            with open('setup.py', 'r') as f:
                content = f.read().lower()
                if 'name="bokeh"' in content or "name='bokeh'" in content:
                    if 'version="2.' in content or "version='2." in content:
                        version_major = 2
                    elif 'version="1.' in content or "version='1." in content:
                        version_major = 1
        elif os.path.exists('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                content = f.read().lower()
                if 'name = "bokeh"' in content:
                    if 'version = "2.' in content or "version = '2." in content:
                        version_major = 2
                    elif 'version = "1.' in content or "version = '1." in content:
                        version_major = 1
        
        # Build the install command based on the version
        if version_major < 3:
            # For older Bokeh versions
            install_cmd = (
                f"uv pip install "
                f'"Jinja2{jinja2_ver}" '
                f'"numpy{numpy_ver}" '
                f'"pandas{pandas_ver}" '
                f'"tornado{tornado_ver}" '
                f'"pillow{pillow_ver}" '
                f'"PyYAML>=3.10" '
                f'"packaging>=16.8" '
                f'"pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"'
            )
        else:
            # For Bokeh 3.x and newer
            install_cmd = (
                f"uv pip install "
                f'"Jinja2{jinja2_ver}" '
                f'"numpy{numpy_ver}" '
                f'"pandas{pandas_ver}" '
                f'"tornado{tornado_ver}" '
                f'"pillow{pillow_ver}" '
                f'"contourpy{contourpy_ver}" '
                f'"PyYAML>=3.10" '
                f'"xyzservices>=2021.09.1" '
                f'"packaging>=16.8" '
                f'"narwhals>=1.13" '
                f'"pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"'
            )
        try:
            run_in_venv(install_cmd, timeout=120)  # 2 minute timeout for dependency install
        except subprocess.TimeoutExpired:
            print("Dependency installation timed out, skipping this combination")
            return False

        # Build bokeh and install in editable mode - shorter timeout
        build_cmd = "uv pip install -e ."
        try:
            run_in_venv(build_cmd, timeout=120)  # 2 minute timeout for build
        except subprocess.TimeoutExpired:
            print("Build timed out, skipping this combination")
            return False

        # Check install - use direct path to Python executable in venv
        check_install_cmd = ".venv/bin/python -c 'import bokeh; print(f\"Bokeh {bokeh.__version__} installed successfully\")'"
        result = log_and_run(check_install_cmd, logfile_path)
        
        with open(logfile_path, "a", encoding="utf-8") as f:
            f.write("BUILD SUCCEEDED\n")
            if result.stdout:
                f.write(f"STDOUT:\n{result.stdout}\n")
        
        print("✅ BUILD SUCCEEDED!")
        return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        with open(logfile_path, "a", encoding="utf-8") as f:
            f.write(f"BUILD FAILED\n{repr(e)}\n")
        print("❌ BUILD FAILED")
        return False
    finally:
        # Clean up virtual environment
        if os.path.exists(venv_dir):
            shutil.rmtree(venv_dir)


def smart_sparse_search(logfile_path, version_tag=None):
    """
    Iterate over Python x Jinja2 x NumPy x Pandas x Tornado x Pillow x Contourpy,
    looking for the first successful build.
    Return a dict with the successful combo if found, or None if none succeeded.
    """
    tested_combos = 0
    max_combos = 5  # Try more combinations for better chance of success
    
    # Determine if this is an older version (Bokeh 1.x or 2.x)
    is_older_version = False
    if version_tag and version_tag[0].isdigit():
        is_older_version = int(version_tag[0]) < 3
    
    # Define default combinations based on Bokeh version
    if is_older_version:
        # For older versions, try Python 3.7 first with older dependency versions
        default_combos = [
            ("3.7", ">=2.9", ">=1.12", ">=0.25.3", ">=5.1", ">=6.0.0", ">=1.0.0"),
            ("3.8", "==2.10.3", "==1.19.5", ">=1.0", ">=6.0", ">=7.1.0", ">=1.0.0"),
        ]
    else:
        # For newer versions (3.x), use newer Python versions
        default_combos = [
            ("3.10", ">=2.9", ">=1.16", ">=1.2", ">=6.2", ">=7.1.0", ">=1.2"),
            ("3.9", ">=2.9", ">=1.16", ">=1.2", ">=6.2", ">=7.1.0", ">=1.2"),
        ]
    
    # Generate additional combinations for more thorough testing
    additional_combos = list(itertools.product(
        PYTHON_VERSIONS,
        JINJA2_VERSIONS,
        NUMPY_VERSIONS,
        PANDAS_VERSIONS,
        TORNADO_VERSIONS,
        PILLOW_VERSIONS,
        CONTOURPY_VERSIONS,
    ))
    
    # Remove any duplicates of the default combos
    filtered_combos = [combo for combo in additional_combos if combo not in default_combos]
    
    # Take just a small subset
    if filtered_combos:
        random.shuffle(filtered_combos)
        filtered_combos = filtered_combos[:max_combos - len(default_combos)]
    
    combos_to_try = default_combos + filtered_combos

    for (
        python_ver,
        jinja2_ver,
        numpy_ver,
        pandas_ver,
        tornado_ver,
        pillow_ver,
        contourpy_ver,
    ) in combos_to_try:
        tested_combos += 1
        combo_str = (
            f"py={python_ver}, jinja2={jinja2_ver}, numpy={numpy_ver}, "
            f"pandas={pandas_ver}, tornado={tornado_ver}, pillow={pillow_ver}, "
            f"contourpy={contourpy_ver}"
        )
        print(f"Trying combo {tested_combos}/{max_combos}: {combo_str}")

        success = build_bokeh_with_versions(
            python_ver,
            jinja2_ver,
            numpy_ver,
            pandas_ver,
            tornado_ver,
            pillow_ver,
            contourpy_ver,
            logfile_path,
        )
        if success:
            print(f"SUCCESS with this combination! Stopping search after {tested_combos} attempts.")
            return {
                "python": python_ver,
                "jinja2": jinja2_ver,
                "numpy": numpy_ver,
                "pandas": pandas_ver,
                "tornado": tornado_ver,
                "pillow": pillow_ver,
                "contourpy": contourpy_ver,
            }

    print(f"\nExhausted {tested_combos} combos; none succeeded.")
    return None


def build_single_version(version_tag):
    """
    Clones the bokeh repo at version_tag into a temporary directory,
    runs the matrix search, and saves the result to a file.
    Returns True if any combo succeeded, otherwise False.
    """
    base_temp_dir = tempfile.mkdtemp(prefix=f"bokeh_{version_tag}_")
    logfile_path = os.path.join(base_temp_dir, f"build_commands_{version_tag}.log")

    print(f"\n[INFO] Building bokeh {version_tag} in {base_temp_dir}")
    try:
        # 1. Clone
        target_dir = os.path.join(base_temp_dir, "bokeh_repo")
        cmd_clone = f"git clone {BOKEH_REPO_URL} {target_dir}"
        log_and_run(cmd_clone, logfile_path)

        # 2. Checkout the specific tag
        cmd_checkout = f"cd {target_dir} && git checkout {version_tag}"
        log_and_run(cmd_checkout, logfile_path)

        # 3. cd into the repo
        os.chdir(target_dir)

        # 4. Run the matrix search with version information
        success_combo = smart_sparse_search(logfile_path, version_tag=version_tag)

        # 5. If a combo succeeded, record it in build_results.txt
        if success_combo is not None:
            combo_line = (
                f"bokeh {version_tag} -> python={success_combo['python']}, "
                f"jinja2={success_combo['jinja2']}, "
                f"numpy={success_combo['numpy']}, "
                f"pandas={success_combo['pandas']}, "
                f"tornado={success_combo['tornado']}, "
                f"pillow={success_combo['pillow']}, "
                f"contourpy={success_combo['contourpy']}\n"
            )
            # Append to the global results file relative to the script's location
            results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(combo_line)
            return True
        else:
            combo_line = f"bokeh {version_tag} -> NO SUCCESS\n"
            results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(combo_line)
            return False

    except Exception as e:
        print(f"Error building bokeh {version_tag}: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))


def build_single_commit(commit_hash):
    """
    Clones the bokeh repo at the specified commit hash into a temporary directory,
    runs the matrix search, and saves the result to a file.
    Returns True if any combo succeeded, otherwise False.
    """
    commit_short = commit_hash[:7]
    base_temp_dir = tempfile.mkdtemp(prefix=f"bokeh_commit_{commit_short}_")
    logfile_path = os.path.join(base_temp_dir, f"build_commands_{commit_short}.log")

    print(f"\n[INFO] Building bokeh commit {commit_short} in {base_temp_dir}")
    try:
        # 1. Clone
        target_dir = os.path.join(base_temp_dir, "bokeh_repo")
        cmd_clone = f"git clone {BOKEH_REPO_URL} {target_dir}"
        log_and_run(cmd_clone, logfile_path)

        # 2. Checkout the specific commit
        cmd_checkout = f"cd {target_dir} && git checkout {commit_hash}"
        log_and_run(cmd_checkout, logfile_path)

        # 3. cd into the repo
        os.chdir(target_dir)

        # 4. Run the matrix search
        success_combo = smart_sparse_search(logfile_path)

        # 5. If a combo succeeded, record it in build_results.txt
        if success_combo is not None:
            combo_line = (
                f"bokeh commit {commit_short} -> python={success_combo['python']}, "
                f"jinja2={success_combo['jinja2']}, "
                f"numpy={success_combo['numpy']}, "
                f"pandas={success_combo['pandas']}, "
                f"tornado={success_combo['tornado']}, "
                f"pillow={success_combo['pillow']}, "
                f"contourpy={success_combo['contourpy']}\n"
            )
            # Append to the global results file relative to the script's location
            results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(combo_line)
            return True
        else:
            combo_line = f"bokeh commit {commit_short} -> NO SUCCESS\n"
            results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(combo_line)
            return False

    except Exception as e:
        print(f"Error building bokeh commit {commit_short}: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_recent_commits(n=3):
    """Get the most recent n commits from the Bokeh repository."""
    temp_dir = tempfile.mkdtemp(prefix="bokeh_get_commits_")
    try:
        cmd_clone = f"git clone {BOKEH_REPO_URL} {temp_dir}"
        subprocess.run(cmd_clone, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        os.chdir(temp_dir)
        result = subprocess.run(
            f"git log -n {n} --pretty=format:'%H'", 
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        commits = result.stdout.strip().split("\n")
        return commits
    finally:
        # Clean up
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(temp_dir)


def main():
    # Clean or create the overall results file
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
    if os.path.exists(results_path):
        os.remove(results_path)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# Build results for bokeh versions and commits:\n")

    # Test with several versions across major releases
    test_versions = TEST_VERSIONS  # Use the list defined at the top of the file
    print(f"Testing with versions: {test_versions}")
    
    success_count = 0
    for version in test_versions:
        result = build_single_version(version)
        if result:
            success_count += 1
            print(f"✅ Successfully built version {version}")
        else:
            print(f"❌ Failed to build version {version}")
    
    if success_count == 0:
        print("No test versions were built successfully. Exiting.")
        return
    
    # Now test with a single recent commit
    print("\nTesting with a recent commit...")
    recent_commits = get_recent_commits(1)
    
    for commit in recent_commits:
        result = build_single_commit(commit)
        if result:
            print(f"✅ Successfully built commit {commit[:7]}")
        else:
            print(f"❌ Failed to build commit {commit[:7]}")
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUILD_RESULTS_FILE)
    print(f"\nBuild results have been saved to {results_path}")
    
    # Full script information
    print("\nTo build all versions and commits, modify the script to include more versions in BOKEH_VERSIONS")
    print("and uncomment the parallel execution section.")


if __name__ == "__main__":
    main()