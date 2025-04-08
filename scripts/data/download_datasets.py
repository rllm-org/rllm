import os
import gdown
import shutil

# Define the Google Drive file IDs for the JSON files
FILE_IDS = {
    "train_taco.json": "1XpZYH9MP7N-YR9mCLCxs9181zedxivOU",
    "olympiad.json": "1TxTUkXR5WIXS1586XbkrkfpKLEyHiNJS",
    "train_livecodebench.json": "1-lKdRfRjytdTltgLyAxTqVRoksI2cJfU",
    "test_livecodebench.json": "1B0sotl48BLd4gqlitL5HVJf1cy3RxpEV",
    "kodcode.json": "1STMAebzGjJtgl5OcOhG-4hjdAiLAWiY6",
    "primeintellect.json": "1o-4P5fUBZd75PM9qfSXInPYXzfW5dfYm",
}

# Define the destination paths
DEST_PATHS = {
    "train_taco.json": os.path.expanduser("~/rllm/rllm/data/train/code/taco.json"),
    "olympiad.json": os.path.expanduser("~/rllm/rllm/data/train/math/olympiad.json"),
    "test_livecodebench.json": os.path.expanduser("~/rllm/rllm/data/test/code/livecodebench.json"),
    "train_livecodebench.json": os.path.expanduser("~/rllm/rllm/data/train/code/livecodebench.json"),
    "kodcode.json": os.path.expanduser("~/rllm/rllm/data/train/code/kodcode.json"),
    "primeintellect.json": os.path.expanduser("~/rllm/rllm/data/train/code/primeintellect.json"),
}

# Create the necessary directories
for path in DEST_PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Download and move files
for filename, file_id in FILE_IDS.items():
    temp_path = f"./{filename}"  # Download location
    dest_path = DEST_PATHS[filename]

    # Download the file
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False)

    # Move to the correct location
    print(f"Moving {filename} to {dest_path}...")
    shutil.move(temp_path, dest_path)

print("All files downloaded and moved successfully.")