#!/bin/bash
set -e

# Bokeh installation script for different versions
# This script can be called with a version parameter to install different Bokeh versions

install_bokeh() {
    local version=$1
    local python_version=$2
    echo "Installing Bokeh version: $version with Python $python_version"

    # Clean up existing venv if present
    if [ -d ".venv" ]; then
        rm -rf .venv
    fi

    # Create virtual environment
    uv venv --python $python_version
    source .venv/bin/activate

    case $version in
        "1.4.0")
            # Install older dependencies for Bokeh 1.x
            uv pip install setuptools wheel
            uv pip install "Jinja2>=2.9" "numpy>=1.12" "pandas>=0.25.3" "tornado>=5.1" "pillow>=6.0.0" "PyYAML>=3.10" "packaging>=16.8"
            uv pip install "pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"
            ;;
        "2.0.0"|"2.1.1"|"2.2.3"|"2.3.3"|"2.4.3")
            # Install dependencies for Bokeh 2.x
            uv pip install setuptools wheel
            uv pip install "Jinja2>=2.9" "numpy>=1.16" "pandas>=1.0" "tornado>=6.0" "pillow>=7.1.0" "PyYAML>=3.10" "packaging>=16.8"
            uv pip install "pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"
            ;;
        *)
            # For Bokeh 3.x and newer versions (default)
            uv pip install "Jinja2>=2.9" "contourpy>=1.2" "narwhals>=1.13" "numpy>=1.16" "packaging>=16.8" "pandas>=1.2" "pillow>=7.1.0" "PyYAML>=3.10" "tornado>=6.2" "xyzservices>=2021.09.1"
            uv pip install "pytest" "pytest-cov" "pytest-xdist" "pytest-asyncio" "pytest-timeout" "colorama"
            ;;
    esac

    # Clone Bokeh repository if needed
    local repo_dir="bokeh_repo"
    if [ ! -d "$repo_dir" ]; then
        git clone https://github.com/bokeh/bokeh.git $repo_dir
    fi

    # Checkout specific version
    (cd $repo_dir && git fetch --all && git checkout $version)

    # Check if this is a version that requires interactive build
    if [[ $(echo "$version" | cut -d. -f1) -lt 3 ]]; then
        # For older versions (before 3.0), we need to use a more direct approach
        
        # First, ensure node.js is available
        if ! command -v npm &> /dev/null; then
            echo "Error: npm is required for building older Bokeh versions. Please install Node.js."
            return 1
        fi
        
        # For older versions, build BokehJS manually first
        (cd $repo_dir/bokehjs && npm ci && npm run build)
        
        # For older Bokeh versions, we need a more direct approach
        # First, check if the bokehjs build directory exists
        if [[ ! -d "$repo_dir/bokehjs/build" ]]; then
            echo "Error: BokehJS build directory not found after building"
            return 1
        fi
        
        # Create a completely new simplified setup script that bypasses all the prompts
        cat > $repo_dir/setup_direct.py << EOF
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
    'version': '2.4.3',  # Hardcoded for this script
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

        # Install the package with our custom setup script
        (cd $repo_dir && python setup_direct.py develop)
        
        # Success message
        echo "Successfully installed Bokeh using custom direct setup"
    else
        # For newer versions (3.0+), no interactive prompt
        (cd $repo_dir && uv pip install -e .)
    fi

    # Verify installation
    python -c "import bokeh; print(f'Successfully installed Bokeh {bokeh.__version__}')"

    echo "✅ Bokeh $version installed successfully!"
}

# Default installation if no arguments provided
if [ $# -eq 0 ]; then
    # Install the latest version with Python 3.10
    install_bokeh "3.4.0" "3.10"
else
    # Parse command line arguments
    case "$1" in
        "1.4.0")
            install_bokeh "1.4.0" "3.7"
            ;;
        "2.0.0")
            install_bokeh "2.0.0" "3.8"
            ;;
        "2.4.3")
            install_bokeh "2.4.3" "3.9"
            ;;
        "3.0.0")
            install_bokeh "3.0.0" "3.10"
            ;;
        "3.4.0")
            install_bokeh "3.4.0" "3.10"
            ;;
        "latest")
            install_bokeh "3.4.0" "3.10"
            ;;
        "commit")
            if [ -z "$2" ]; then
                echo "Error: Please provide a commit hash when using the 'commit' option"
                exit 1
            fi
            install_bokeh "$2" "3.10"
            ;;
        *)
            echo "Unknown version: $1"
            echo "Available options: 1.4.0, 2.0.0, 2.4.3, 3.0.0, 3.4.0, latest, commit <hash>"
            exit 1
            ;;
    esac
fi

echo "Bokeh installation complete"