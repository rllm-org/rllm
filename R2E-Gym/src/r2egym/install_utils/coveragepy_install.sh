set -e

check_install() {
    echo "Verifying installation..."
    if python -c "import coverage; print('CoveragePy version:', coverage.__version__)"; then
        echo "✅ Installation successful!"
        return 0
    else
        echo "❌ Verification failed"
        return 1
    fi
}

test_39_install () {
    uv venv --python 3.7
    source .venv/bin/activate

    uv pip install -r requirements/dev.pip
    uv pip install setuptools pytest 
    uv pip install -e .
    uv run python igor.py zip_mods

    check_install
}


test_37_install () {
    uv venv --python 3.10
    source .venv/bin/activate

    uv pip install -r requirements/dev.pip
    uv pip install setuptools pytest 
    uv pip install -e .
    uv run python igor.py zip_mods

    check_install
}


echo "Starting CoveragePy installation attempts..."

# Try Python 3.9 installation
if test_39_install; then
    echo "Successfully installed CoveragePy using Python 3.9"
    exit 0
fi

echo "Python 3.9 installation failed, trying Python 3.7..."

# Try Python 3.7 installation
if test_37_install; then
    echo "Successfully installed CoveragePy using Python 3.7"
    exit 0
fi