#!/bin/bash

set -e  # Exit on any error

check_pillow() {
    echo "Verifying Pillow installation..."
    if python -c "import PIL; from PIL import Image; Image.new('RGB', (1, 1))" &> /dev/null; then
        echo "✅ Pillow installation successful!"
        return 0
    else
        echo "❌ Pillow verification failed"
        return 1
    fi
}

main() {
    echo "Starting Pillow installation attempts..."
    
    uv venv --python 3.9
    source .venv/bin/activate
    uv pip install setuptools pytest pytest-cov PyQt5
    uv pip install -e . --no-build-isolation

    if check_pillow; then
        echo "Successfully installed Pillow"
        return 0
    fi

    return 1
}

# Run the main function
main