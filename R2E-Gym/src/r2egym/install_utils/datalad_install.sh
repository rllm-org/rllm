
set -e

check_install() {
    echo "Verifying installation..."
    if python -c "import datalad; print('Datalad version:', datalad.__version__)"; then
        echo "✅ Installation successful!"
        return 0
    else
        echo "❌ Verification failed"
        return 1
    fi
}

test_39_install () {
    uv venv --python 3.9
    source .venv/bin/activate

    uv pip install setuptools pytest pytest-cov numpy 'pybids<0.7.0'
    uv pip install -e .[full]
    uv pip install -e .[devel]

    check_install
}


test_37_install () {
    uv venv --python 3.7
    source .venv/bin/activate

    uv pip install setuptools pytest pytest-cov numpy 'pybids<0.7.0' fasteners bids
    uv pip install -e .[full]
    uv pip install -e .[devel]

    check_install
}


echo "Starting Datalad installation attempts..."

# Try Python 3.9 installation
if test_39_install; then
    echo "Successfully installed Datalad using Python 3.9"
    exit 0
fi

echo "Python 3.9 installation failed, trying Python 3.7..."

# Try Python 3.7 installation
if test_37_install; then
    echo "Successfully installed Datalad using Python 3.7"
    exit 0
fi