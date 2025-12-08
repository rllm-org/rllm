import os


def transform_file(filepath: str) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple string replacements
    new_content = content.replace("asyncio.async(", "asyncio.create_task(")

    if new_content != content:
        print(f"Transforming {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def process_directory(directory: str) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    transform_file(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    # Process current directory
    directory = "aiohttp"  # or specify your directory path
    process_directory(directory)
