#!/bin/bash
set -e

# Script to install various Bokeh versions based on our testing results
# This script will install multiple versions of Bokeh in separate directories

WORK_DIR="/tmp/bokeh_versions"
mkdir -p "$WORK_DIR"

# Function to install a specific Bokeh version
install_bokeh_version() {
  local version=$1
  local python_version=$2
  local install_dir="$WORK_DIR/bokeh_$version"
  
  echo "============================================="
  echo "Installing Bokeh $version with Python $python_version"
  echo "============================================="
  
  # Create installation directory
  mkdir -p "$install_dir"
  cd "$install_dir"
  
  # Clone repository
  if [ ! -d "repo" ]; then
    git clone https://github.com/bokeh/bokeh.git repo
  fi
  
  # Clean previous virtual environment if it exists
  if [ -d ".venv" ]; then
    rm -rf .venv
  fi
  
  # Create and activate virtual environment
  uv venv --python="$python_version"
  source .venv/bin/activate
  
  # Install dependencies based on version
  cd repo
  git fetch --all
  git checkout "$version"
  
  # Install appropriate dependencies based on version
  local major_version=$(echo "$version" | cut -c1-1)
  
  if [ "$major_version" = "1" ]; then
    # Bokeh 1.x dependencies
    uv pip install setuptools wheel
    uv pip install "Jinja2>=2.9" "numpy>=1.12" "pandas>=0.25.3" "tornado>=5.1" "pillow>=6.0.0" "PyYAML>=3.10" "packaging>=16.8"
    uv pip install "pytest" "pytest-cov" "pytest-xdist"
  elif [ "$major_version" = "2" ]; then
    # Bokeh 2.x dependencies
    uv pip install setuptools wheel
    uv pip install "Jinja2>=2.9" "numpy>=1.16" "pandas>=1.0" "tornado>=6.0" "pillow>=7.1.0" "PyYAML>=3.10" "packaging>=16.8"
    uv pip install "pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio"
  else
    # Bokeh 3.x dependencies
    uv pip install "Jinja2>=2.9" "contourpy>=1.2" "narwhals>=1.13" "numpy>=1.16" "packaging>=16.8" "pandas>=1.2" "pillow>=7.1.0" "PyYAML>=3.10" "tornado>=6.2" "xyzservices>=2021.09.1"
    uv pip install "pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"
  fi
  
  # Install Bokeh - handle interactive prompt for older versions
  if [[ $(echo "$version" | cut -d. -f1) -lt 3 ]]; then
    # For older versions (before 3.0), we need to use a more direct approach
    
    # First, ensure node.js is available
    if ! command -v npm &> /dev/null; then
        echo "Error: npm is required for building older Bokeh versions. Please install Node.js."
        return 1
    fi
    
    # For older versions, build BokehJS manually first
    (cd bokehjs && npm ci && npm run build)
    
    # First, check if the bokehjs build directory exists
    if [[ ! -d "bokehjs/build" ]]; then
        echo "Error: BokehJS build directory not found after building"
        return 1
    fi
    
    # Create a completely new simplified setup script that bypasses all the prompts
    cat > setup_direct.py << EOF
# Direct setup script that doesn't use the built-in _setup_support.py module
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

# Basic setup info
setup_args = {
    'name': 'bokeh',
    'version': '$version',  # Version from environment
    'packages': find_packages(where=".", exclude=["scripts*", "tests*"]),
    'package_data': package_data,
    'install_requires': [
        'Jinja2>=2.9',
        'numpy>=1.11.3',
        'packaging>=16.8',
        'pillow>=7.1.0',
        'PyYAML>=3.10',
        'tornado>=5.1',
        'typing_extensions>=3.10.0',
    ],
    'python_requires': '>=3.7',
}

setup(**setup_args)
EOF
    
    # Then install the Python package with our modified setup script
    python setup_direct.py develop
    
    # Success message
    echo "Successfully installed Bokeh using custom direct setup"
  else
    # For newer versions (3.0+), no interactive prompt
    uv pip install -e .
  fi
  
  # Verify installation
  python -c "import bokeh, platform; print(f'Successfully installed Bokeh {bokeh.__version__} with Python {platform.python_version()}')"
  
  # Create a shell script in the version directory that activates the environment
  cat > "$install_dir/activate_bokeh.sh" << EOL
#!/bin/bash
# Script to activate Bokeh $version environment
echo "Activating Bokeh $version with Python $python_version"
source "$install_dir/.venv/bin/activate"
cd "$install_dir/repo"
echo "Environment activated. You can now use Bokeh $version with Python $python_version"
EOL
  
  chmod +x "$install_dir/activate_bokeh.sh"
  
  echo "✅ Bokeh $version installed successfully in $install_dir"
  echo "To activate this environment later, run: source $install_dir/activate_bokeh.sh"
  echo ""
  
  # Deactivate virtual environment
  deactivate
}

# Install multiple Bokeh versions
echo "Installing multiple versions of Bokeh..."

# Older versions
install_bokeh_version "1.4.0" "3.7"
install_bokeh_version "2.0.0" "3.8" 
install_bokeh_version "2.4.3" "3.9"

# Newer versions
install_bokeh_version "3.0.0" "3.10"
install_bokeh_version "3.4.0" "3.10"

echo "==================================================="
echo "All Bokeh versions have been installed successfully!"
echo "==================================================="
echo "Installed versions:"
echo " - Bokeh 1.4.0 (Python 3.7) - $WORK_DIR/bokeh_1.4.0"
echo " - Bokeh 2.0.0 (Python 3.8) - $WORK_DIR/bokeh_2.0.0"
echo " - Bokeh 2.4.3 (Python 3.9) - $WORK_DIR/bokeh_2.4.3"
echo " - Bokeh 3.0.0 (Python 3.10) - $WORK_DIR/bokeh_3.0.0"
echo " - Bokeh 3.4.0 (Python 3.10) - $WORK_DIR/bokeh_3.4.0"
echo ""
echo "To activate a specific version, run:"
echo "source $WORK_DIR/bokeh_<version>/activate_bokeh.sh"