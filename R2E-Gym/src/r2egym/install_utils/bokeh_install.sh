#!/usr/bin/env bash
#
# bokeh_install.sh
#
# Tries three different approaches to build and install Bokeh from the current directory.
# Exits on the first approach that succeeds.

# Stop on any error
set -e

########################
# Helper function to check if Bokeh was imported successfully
########################
check_bokeh_import() {
  # IMPORTANT: Return 1 if import fails, so the function signals failure
  if ! .venv/bin/python -c "import bokeh; print('Bokeh version:', bokeh.__version__); print('Successfully imported Bokeh')"; then
    echo "[ERROR] Bokeh import failed!"
    return 1
  fi
  return 0
}

########################
# Helper function to install Bokeh
########################
install_bokeh() {
  local python_ver="$1"
  local extra_deps="$2"
  local use_custom_setup="$3"
  
  echo ""
  echo "[INFO] Creating new virtual environment with Python ${python_ver} ..."
  rm -rf .venv
  uv venv --python "${python_ver}"
  
  # Activate the new environment
  source .venv/bin/activate
  
  echo "[INFO] Upgrading pip and wheel ..."
  uv pip install --upgrade pip wheel setuptools
  
  echo "[INFO] Installing core dependencies ..."
  uv pip install "Jinja2>=2.9" "numpy" "packaging>=16.8" "pillow" "PyYAML>=3.10" "tornado" "six" "flask"
  
  # Install extra dependencies if provided
  if [ -n "$extra_deps" ]; then
    echo "[INFO] Installing extra dependencies: $extra_deps ..."
    uv pip install $extra_deps
  fi
  
  echo "[INFO] Installing test dependencies ..."
  uv pip install pytest pytest-cov pytest-xdist pytest-asyncio
  
  # For older Bokeh versions, build BokehJS manually
  if [ -d "bokehjs" ]; then
    echo "[INFO] Building BokehJS manually ..."
    if ! command -v npm &> /dev/null; then
      echo "[WARNING] npm not found, this may cause installation to fail for older Bokeh versions"
    else
      # For older versions that don't have package-lock.json, use npm install instead of npm ci
      if [ -f "bokehjs/package-lock.json" ]; then
        (cd bokehjs && npm ci && npm run build) || echo "[WARNING] BokehJS build failed with npm ci, trying npm install..."
      fi
      
      # If npm ci failed or no package-lock.json exists, try npm install
      if [ ! -d "bokehjs/build" ]; then
        (cd bokehjs && npm install && npm run build) || echo "[WARNING] BokehJS build failed"
      fi
    fi
  fi
  
  if [ "$use_custom_setup" = "true" ] && [ -d "bokehjs" ]; then
    echo "[INFO] Using custom setup script to bypass interactive prompts ..."
    
    # Create a custom setup script
    cat > setup_direct.py << 'EOF'
# Direct setup script that bypasses prompts
import setuptools
from setuptools import setup, find_packages
import os
import sys

# Important paths
bokehjs_dir = os.path.join(os.path.dirname(__file__), "bokehjs")
bokehjs_build_dir = os.path.join(bokehjs_dir, "build")

# Common package data to include
package_data = {
    'bokeh': [
        'py.typed',
        'core/_templates/*.html',
        'core/_templates/*.js',
        'core/_templates/*.css',
        'core/script.js',
        'sphinxext/_templates/*.html',
        'sphinxext/_templates/*.js',
        'themes/*.yaml',
        'themes/*.json',
        'util/*.js',
        'util/*.css',
    ]
}

# If there is already a JS build then use it
if os.path.exists(bokehjs_build_dir):
    package_data['bokeh'] += [
        'server/static/*.js',
        'server/static/*.css',
        'server/static/*.map',
        'server/static/js/*.js',
        'server/static/js/*.js.map',
        'server/static/js/compiler/*.js',
        'server/static/js/compiler/*.js.map',
        'server/static/js/bokehjs/*.js',
        'server/static/js/bokehjs/*.js.map',
        'server/static/js/lib/*.js',
        'server/static/js/lib/*.js.map',
        'server/static/js/types/*.js',
        'server/static/js/types/*.js.map',
    ]

# Try to extract version
version = "0.0.0"  # fallback
try:
    import re
    for version_file in ["_version.py", "version.py", "__init__.py"]:
        possible_path = os.path.join("bokeh", version_file)
        if os.path.exists(possible_path):
            with open(possible_path, "r") as f:
                content = f.read()
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    version = match.group(1)
                    break
except Exception as e:
    print(f"Error extracting version: {e}")
    pass

# Basic setup info
setup_args = {
    'name': 'bokeh',
    'version': version,
    'packages': find_packages(where=".", exclude=["scripts*", "tests*"]),
    'package_data': package_data,
    'install_requires': [
        'Jinja2>=2.9',
        'numpy>=1.11.3',
        'packaging>=16.8',
        'pillow>=7.1.0',
        'PyYAML>=3.10',
        'tornado>=5.1',
        'six',  # Required for older Bokeh versions (0.8.0)
        'flask',  # Required for older Bokeh versions (0.8.0)
    ],
    'python_requires': '>=3.7',
}

setup(**setup_args)
EOF
    
    # Install with custom setup script
    python setup_direct.py develop
  else
    echo "[INFO] Installing Bokeh with standard setup ..."
    # Use regular pip install
    uv pip install -e .
  fi
  
  # Check if import works
  if ! check_bokeh_import; then
    echo "[ERROR] Bokeh import check failed! Installing additional dependencies and retrying..."
    # Try installing some additional dependencies that might be needed for older versions
    uv pip install "colorama" "pyyaml" "requests" "markdown" "pygments"
    
    # Check again
    if ! check_bokeh_import; then
      return 1
    fi
  fi
  
  echo "[SUCCESS] Build and import succeeded with Python=${python_ver}"
  return 0
}

########################
# Attempt #1: Python 3.7 for Old Bokeh (0.x and 1.x)
########################
echo "[Attempt #1] Trying Python=3.7 with dependencies for older Bokeh versions"
if install_bokeh "3.7" "pandas<1.0 pytest<7.0 six flask" "true"; then
  echo "[INFO] First approach succeeded. Exiting."
  exit 0
fi

########################
# Attempt #2: Python 3.9 for Bokeh 2.x
########################
echo "[Attempt #2] Trying Python=3.9 with dependencies for Bokeh 2.x"
if install_bokeh "3.9" "pandas>=1.0 six flask" "true"; then
  echo "[INFO] Second approach succeeded. Exiting."
  exit 0
fi

########################
# Attempt #3: Python 3.10 for Bokeh 3.x
########################
echo "[Attempt #3] Trying Python=3.10 with dependencies for Bokeh 3.x"
if install_bokeh "3.10" "contourpy>=1.2 narwhals>=1.13 pandas>=1.2 xyzservices>=2021.09.1" "false"; then
  echo "[INFO] Third approach succeeded. Exiting."
  exit 0
fi

########################
# If none succeeded
########################
echo "[ERROR] All three installation attempts failed."
exit 1