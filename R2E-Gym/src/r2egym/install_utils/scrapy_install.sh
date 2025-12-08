uv venv --python=python3.9
source .venv/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt file. Installing dependencies..."
    
    # Check if pip is installed
    uv pip install -r requirements.txt
    
    echo "Dependencies installation completed!"
else
    echo "No requirements.txt file found in the current directory."
fi

uv pip install -e .

uv pip install pytest testfixtures pyftpdlib pexpect