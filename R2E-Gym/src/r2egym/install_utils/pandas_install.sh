#!/usr/bin/env bash
#
# build_pandas_three_combos.sh
#
# Tries three different sets of pinned Python/NumPy/Cython/etc. versions
# to build (and import) pandas. Exits on the first combination that succeeds.

# Stop on any error
set -e

########################
# Setup the "versioneer" command macro
########################
VERSIONEER_COMMAND='echo -e "[versioneer]\nVCS = git\nstyle = pep440\nversionfile_source = pandas/_version.py\nversionfile_build = pandas/_version.py\ntag_prefix =\nparentdir_prefix = pandas-" > setup.cfg && versioneer install'

########################
# Helper function to build & check
########################
build_and_check_pandas() {
  local python_ver="$1"
  local numpy_expr="$2"
  local cython_expr="$3"
  local setuptools_expr="$4"
  local versioneer_expr="$5"

  echo ""
  echo "[INFO] Creating new virtual environment with Python ${python_ver} ..."
  uv venv --python "${python_ver}"

  # Activate the new environment
  source .venv/bin/activate

  echo "[INFO] Upgrading pip and wheel ..."
  uv pip install --upgrade pip wheel

  echo "[INFO] Installing pinned dependencies ..."
  uv pip install --upgrade \
    "setuptools==${setuptools_expr}" \
    "numpy==${numpy_expr}" \
    "cython${cython_expr}" \
    "versioneer==${versioneer_expr}" \
    python-dateutil pytz pytest hypothesis jinja2

  uv pip install -r requirements-dev.txt

  echo "[INFO] Running versioneer setup ..."
  # The versioneer script is placed inline:
  bash -c "set -e; source .venv/bin/activate && ${VERSIONEER_COMMAND}"

  echo "[INFO] Removing pyproject.toml if present (for older builds) ..."
  rm -f pyproject.toml

  echo "[INFO] Cleaning pandas build ..."
  uv run python setup.py clean --all

  echo "[INFO] Building pandas with CFLAGS='-O0 -Wno-error=array-bounds' ..."
  # Using uv run for the build:
  CFLAGS="-O0 -Wno-error=array-bounds" uv run python setup.py build_ext --inplace -j 4

  echo "[INFO] Installing pandas in editable mode ..."
  uv run pip install -e .

  echo "[INFO] Checking import of pandas ..."
  
  # IMPORTANT: Return 1 if import fails, so the function signals failure
  if ! .venv/bin/python -c "import pandas; print('Pandas version:', pandas.__version__); print(pandas.DataFrame([[1,2,3]]))"; then
    echo "[ERROR] Pandas import failed!"
    return 1
  fi

  echo "[SUCCESS] Build and import succeeded with Python=${python_ver}, NumPy=${numpy_expr}, Cython${cython_expr}."
}

########################
# Attempt #1
########################
echo "[Attempt #1] Trying Python=3.7, NumPy=1.17.*, Cython<0.30, setuptools=62.*, versioneer=0.23"
if build_and_check_pandas "3.7" "1.17.*" "<0.30" "62.*" "0.23"; then
  echo "[INFO] First combo succeeded. Exiting."
  exit 0
fi

########################
# Attempt #2
########################
echo "[Attempt #2] Trying Python=3.8, NumPy=1.20.*, Cython<0.30, setuptools=62.*, versioneer=0.23"
if build_and_check_pandas "3.8" "1.20.*" "<0.30" "62.*" "0.23"; then
  echo "[INFO] Second combo succeeded. Exiting."
  exit 0
fi

########################
# Attempt #3
########################
echo "[Attempt #3] Trying Python=3.10, NumPy=1.26.*, Cython===3.0.5, setuptools=62.*, versioneer=0.23"
if build_and_check_pandas "3.10" "1.26.*" "===3.0.5" "62.*" "0.23"; then
  echo "[INFO] Third combo succeeded. Exiting."
  exit 0
fi

########################
# If none succeeded
########################
echo "[ERROR] All three attempts failed."
exit 1
--------------------------------------------------------------------------------

In particular, note the change in the import check:  
if ! .venv/bin/python -c "import pandas; ..."; then  
    echo "[ERROR] Pandas import failed!"  
    return 1  
fi  
