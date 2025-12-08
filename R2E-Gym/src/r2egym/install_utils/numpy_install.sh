#!/bin/bash

set -e  # Exit on any error

check_numpy() {
    echo "Verifying NumPy installation..."
    if .venv/bin/python -c "import numpy; numpy.array([1,2])" &> /dev/null; then
        echo "✅ NumPy installation successful!"
        return 0
    else
        echo "❌ NumPy verification failed"
        return 1
    fi
}


try_install_python37() {
    echo "Attempting installation with Python 3.7..."
    uv venv --python 3.7 --python-preference only-managed
    source .venv/bin/activate
    uv pip install "setuptools<=59.8.0" "cython<0.30" pytest pytest-env hypothesis nose
    .venv/bin/python setup.py build_ext --inplace
    check_numpy
}

try_install_python310() {
    echo "Attempting installation with Python 3.10..."
    uv venv --python 3.10 --python-preference only-managed
    source .venv/bin/activate
    uv pip install "setuptools<=59.8.0" "cython<0.30" pytest pytest-env hypothesis nose
    .venv/bin/python setup.py build_ext --inplace
    check_numpy
}

main() {
    echo "Starting NumPy installation attempts..."
    
    # Try Python 3.7 installation
    if try_install_python37; then
        echo "Successfully installed NumPy using Python 3.7"
        return 0
    fi
    
    echo "Python 3.7 installation failed, trying Python 3.10..."
    
    # Try Python 3.11 installation
    if try_install_python310; then
        echo "Successfully installed NumPy using Python 3.10"
        return 0
    fi
    
    echo "All installation attempts failed"
    return 1
}

# Run the main function
main