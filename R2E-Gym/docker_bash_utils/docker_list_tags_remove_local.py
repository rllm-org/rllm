#!/usr/bin/env python3

import subprocess
import requests
import sys
from typing import List, Tuple
import json


def get_local_images(repository: str) -> List[Tuple[str, str]]:
    """Get local Docker images and their tags"""
    cmd = f"docker images {repository} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    images = []
    for line in result.stdout.strip().split("\n"):
        if line:
            repo, tag = line.split(":")
            if tag != "<none>":  # Skip untagged images
                images.append((repo, tag))
    return images


def check_image_in_registry(repository: str, tag: str) -> bool:
    """Check if image exists in Docker Hub"""
    url = f"https://hub.docker.com/v2/repositories/{repository}/tags/{tag}"
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        print(f"Error checking registry for {repository}:{tag}")
        return False


def delete_local_image(repository: str, tag: str) -> bool:
    """Delete local Docker image"""
    cmd = f"docker rmi {repository}:{tag}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error deleting {repository}:{tag}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <repository>")
        print("Example: script.py namanjain12/pandasnew")
        sys.exit(1)

    repository = sys.argv[1]
    local_images = get_local_images(repository)

    if not local_images:
        print(f"No local images found for {repository}")
        return

    print(f"Checking {len(local_images)} local images...")

    deleted_count = 0
    for repo, tag in local_images:
        if check_image_in_registry(repo, tag):
            print(f"Found {repo}:{tag} in registry, deleting locally...")
            if delete_local_image(repo, tag):
                deleted_count += 1
        else:
            print(f"Image {repo}:{tag} not found in registry, keeping locally")

    print(f"\nDeleted {deleted_count} local images that were found in registry")


if __name__ == "__main__":
    main()
