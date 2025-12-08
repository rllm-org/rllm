import itertools
import os
import subprocess
import sys
import tempfile
import shutil
import concurrent.futures

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------
PANDAS_REPO_URL = "https://github.com/pandas-dev/pandas.git"

# Which pandas versions/tags to build:
PANDAS_VERSIONS = [
    "v0.18.0",
    # "v0.18.1",
    # "v0.19.0",
    # "v0.19.1",
    # "v0.19.2",
    # "v0.20.0",
    # "v0.20.1",
    # "v0.20.2",
    # "v0.20.3",
    # "v0.21.0",
    # "v0.21.1",
    # "v0.22.0",
    "v0.23.0",
    # "v0.23.1",
    # "v0.23.2",
    # "v0.23.3",
    # "v0.23.4",
    # "v0.24.0",
    # "v0.24.1",
    # "v0.24.2",
    "v0.25.0",
    # "v0.25.1",
    # "v0.25.2",
    # "v0.25.3",
    "v1.0.0",
    # "v1.0.1",
    # "v1.0.2",
    # "v1.0.3",
    # "v1.0.4",
    # "v1.0.5",
    # "v1.1.0",
    # "v1.1.1",
    # "v1.1.2",
    # "v1.1.3",
    # "v1.1.4",
    # "v1.1.5",
    "v1.2.0",
    # "v1.2.1",
    # "v1.2.2",
    # "v1.2.3",
    # "v1.2.4",
    # "v1.2.5",
    "v1.3.0",
    # "v1.3.1",
    # "v1.3.2",
    # "v1.3.3",
    # "v1.3.4",
    # "v1.3.5",
    # "v1.4.0",
    # "v1.4.1",
    # "v1.4.2",
    # "v1.4.3",
    # "v1.4.4",
    "v1.5.0",
    # "v1.5.1",
    # "v1.5.2",
    # "v1.5.3",
    "v2.0.0",
    # "v2.0.1",
    # "v2.0.2",
    # "v2.0.3",
    # "v2.1.0",
    # "v2.1.1",
    # "v2.1.2",
    # "v2.1.3",
    # "v2.1.4",
    "v2.2.0",
    # "v2.2.1",
    "v2.2.2",
    # "v2.2.3",
]
# PANDAS_VERSIONS = ["v1.3.1"]

PYTHON_VERSIONS = ["3.10", "3.8", "3.7"]  # ["3.7", "3.8", "3.10"]  # , "3.10"]
NUMPY_VERSIONS = [
    # "1.13.*",
    # "1.14.*",
    # "1.15.*",
    # "1.16.*",
    "1.17.*",
    # "1.18.*",
    # "1.19.*",
    "1.20.*",
    # "1.21.*",
    # "1.22.*",
    # "1.23.*",
    # "1.24.*",
    # "1.25.*",
    "1.26.*",
    # "1.27.*",
]
CYTHON_VERSIONS = [
    "<=0.30",
    "==3.0.5",
    # "==0.27.3",
    # "==0.28.5",
    # "==0.26.1",
    # "==0.29.16",
    # "==0.29.21",
    # "==0.29.28",
    # "==0.29.33",
    # "==0.29.36",
]
SETUPTOOLS_VERSIONS = ["62.*"]
VERSIONEER_VERSIONS = ["0.23"]

# This command sets up versioneer in your cloned pandas repo
VERSIONEER_COMMAND = (
    'echo -e "[versioneer]\\nVCS = git\\nstyle = pep440\\n'
    "versionfile_source = pandas/_version.py\\n"
    "versionfile_build = pandas/_version.py\\n"
    'tag_prefix =\\nparentdir_prefix = pandas-" > setup.cfg && '
    "versioneer install"
)

# Where to store overall build results:
BUILD_RESULTS_FILE = "build_results.txt"

# ------------------------------------------------------------------------------
#  LOGGING UTILS
# ------------------------------------------------------------------------------


def log_and_run(cmd, logfile_path, *, shell=True, check=True):
    """
    Logs the command to console and also appends it to logfile_path, then runs it.
    """
    print(f"[RUN] {cmd}")
    with open(logfile_path, "a", encoding="utf-8") as f:
        f.write(f"{cmd}\n")
    return subprocess.run(cmd, shell=shell, check=check, timeout=600)


# ------------------------------------------------------------------------------
#  BUILD LOGIC
# ------------------------------------------------------------------------------


def build_pandas_with_versions(
    python_ver, numpy_ver, cython_ver, setuptools_ver, versioneer_ver, logfile_path
):
    """
    Attempt to build the local checkout of pandas using a new .venv with
    specified versions of Python, NumPy, Cython, setuptools, and versioneer.
    Returns True if the build succeeds, False otherwise.
    """

    try:
        # Create .venv with the given Python version (using 'uv' per your snippet).
        cmd_create_venv = f"uv venv --python {python_ver}"
        log_and_run(cmd_create_venv, logfile_path)

        # We'll assume .venv is created in the current directory:
        activate_path = os.path.join(".venv", "bin", "activate")

        def run_in_venv(cmd):
            full_cmd = f"bash -c 'set -e; source \"{activate_path}\" && {cmd}'"
            log_and_run(full_cmd, logfile_path)

        # Upgrade pip/wheel inside the venv
        run_in_venv(
            "uv pip install --upgrade pip wheel",
            # logfile_path,
        )

        # Install pinned dependencies:
        install_cmd = (
            # "source .venv/bin/activate && "
            f"uv pip install --upgrade "
            f'"setuptools=={setuptools_ver}" '
            f'"numpy=={numpy_ver}" '
            f'"cython {cython_ver}" '
            f'"versioneer=={versioneer_ver}" '
            "python-dateutil pytz pytest hypothesis"
        )
        run_in_venv(install_cmd)  # , logfile_path)

        # Set up versioneer:
        run_in_venv(VERSIONEER_COMMAND)

        # Build pandas (remove pyproject.toml if needed for older builds):
        build_cmd = (
            f'source "{activate_path}" && '
            "rm -f pyproject.toml && "
            'CFLAGS="-O0 -Wno-error=array-bounds" '
            "uv run python setup.py build_ext --inplace -j 4 && "
            "uv run pip install -e ."
        )
        full_cmd = f"bash -c 'set -e; {build_cmd}'"
        log_and_run(full_cmd, logfile_path)

        # Check install
        check_install_cmd = ".venv/bin/python -c 'import pandas; print(pandas.__version__); pandas.DataFrame([[1,2,3]])'"
        log_and_run(check_install_cmd, logfile_path)

        return True

    except subprocess.CalledProcessError as e:
        with open(logfile_path, "a", encoding="utf-8") as f:
            f.write(f"BUILD FAILED\n{repr(e)}")
        return False


import numpy as np

product = itertools.product(
    PYTHON_VERSIONS,
    NUMPY_VERSIONS,
    CYTHON_VERSIONS,
    SETUPTOOLS_VERSIONS,
    VERSIONEER_VERSIONS,
)
product_random = np.random.permutation(list(product))


def smart_sparse_search(logfile_path):
    """
    Iterate over Python x NumPy x Cython x setuptools x versioneer,
    looking for the first successful build.
    Return a dict with the successful combo if found, or None if none succeeded.
    """
    tested_combos = 0

    for (
        python_ver,
        numpy_ver,
        cython_ver,
        setuptools_ver,
        versioneer_ver,
    ) in itertools.product(
        PYTHON_VERSIONS,
        NUMPY_VERSIONS,
        CYTHON_VERSIONS,
        SETUPTOOLS_VERSIONS,
        VERSIONEER_VERSIONS,
    ):
        tested_combos += 1
        combo_str = (
            f"py={python_ver}, numpy={numpy_ver}, cython={cython_ver}, "
            f"setuptools={setuptools_ver}, versioneer={versioneer_ver}"
        )
        print(f"Trying combo: {combo_str}")

        success = build_pandas_with_versions(
            python_ver,
            numpy_ver,
            cython_ver,
            setuptools_ver,
            versioneer_ver,
            logfile_path,
        )
        if success:
            print("SUCCESS with this combination! Stopping search.")
            return {
                "python": python_ver,
                "numpy": numpy_ver,
                "cython": cython_ver,
                "setuptools": setuptools_ver,
                "versioneer": versioneer_ver,
            }

    print(f"\nExhausted {tested_combos} combos; none succeeded.")
    return None


def build_single_version(version_tag):
    """
    Clones the pandas repo at version_tag into a temporary directory,
    runs the matrix search, and saves the result to a file.
    Returns True if any combo succeeded, otherwise False.
    """
    base_temp_dir = tempfile.mkdtemp(prefix=f"pandas_{version_tag}_")
    logfile_path = os.path.join(base_temp_dir, f"build_commands_{version_tag}.log")

    print(f"\n[INFO] Building pandas {version_tag} in {base_temp_dir}")
    try:
        # 1. Clone
        target_dir = os.path.join(base_temp_dir, "pandas_repo")
        cmd_clone = f"git clone {PANDAS_REPO_URL} {target_dir}"
        log_and_run(cmd_clone, logfile_path)

        # 2. Checkout the specific tag
        cmd_checkout = f"cd {target_dir} && git checkout {version_tag}"
        log_and_run(cmd_checkout, logfile_path)

        # 3. cd into the repo
        os.chdir(target_dir)

        # 4. Run the matrix search
        success_combo = smart_sparse_search(logfile_path)

        # Move back to the base temp dir
        os.chdir(base_temp_dir)

        # 5. If a combo succeeded, record it in build_results.txt
        if success_combo is not None:
            combo_line = (
                f"pandas {version_tag} -> python={success_combo['python']}, "
                f"numpy={success_combo['numpy']}, cython={success_combo['cython']}, "
                f"setuptools={success_combo['setuptools']}, "
                f"versioneer={success_combo['versioneer']}\n"
            )
            # Append to a global results file
            with open(BUILD_RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(combo_line)

            return True
        else:
            combo_line = f"pandas {version_tag} -> NO SUCCESS\n"
            with open(BUILD_RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(combo_line)
            return False

    finally:
        # You can comment this out if you want to inspect the files
        # shutil.rmtree(base_temp_dir, ignore_errors=True)
        pass


def main():
    import concurrent.futures

    # Clean or create the overall results file
    if os.path.exists(BUILD_RESULTS_FILE):
        os.remove(BUILD_RESULTS_FILE)
    with open(BUILD_RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("# Build results for all pandas tags:\n")

    # We can run up to 60 builds in parallel
    max_workers = 30

    # Build all requested versions in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for version_tag in PANDAS_VERSIONS:
            # Submit a new process for each version
            fut = executor.submit(build_single_version, version_tag)
            future_map[fut] = version_tag

        # As each future completes, print the result
        for fut in concurrent.futures.as_completed(future_map):
            version_tag = future_map[fut]
            try:
                success = fut.result()
                if success:
                    print(
                        f"[RESULT] Pandas {version_tag} built successfully! (logged in {BUILD_RESULTS_FILE})"
                    )
                else:
                    print(
                        f"[RESULT] Pandas {version_tag} failed every combo. (see {BUILD_RESULTS_FILE})"
                    )
            except Exception as e:
                print(f"[ERROR] Pandas {version_tag} encountered an exception: {e}")


if __name__ == "__main__":
    main()
