import requests
import argparse
from typing import List, Dict, Optional
import sys
from urllib.parse import quote


def fetch_docker_tags(image_name: str) -> List[Dict]:
    """
    Fetch all tags for a Docker image from Docker Hub

    Args:
        image_name: Name of the Docker image (e.g., 'ubuntu', 'nginx')
        limit: Maximum number of tags to return (None for all)

    Returns:
        List of dictionaries containing tag information
    """
    # Handle official images differently

    image_name = quote(image_name)
    base_url = (
        f"https://hub.docker.com/v2/repositories/{image_name}/tags?page_size=50&page="
    )
    tags = []

    for page_idx in range(1, 1000):
        url = base_url + str(page_idx)
        response = requests.get(url)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"{url} not found", file=sys.stderr)
            else:
                print(f"Error fetching tags: {e}", file=sys.stderr)
            return tags
        except requests.exceptions.RequestException as e:
            print(f"Error fetching tags: {e}", file=sys.stderr)
            return tags

        data = response.json()
        tags.extend(data["results"])

    return tags


def format_tags(tags: List[Dict]) -> str:
    """Format tag information for display"""
    output = []
    for tag in tags:
        size = tag["full_size"] / (1024 * 1024)  # Convert to MB
        last_updated = tag["last_updated"].split("T")[0]  # Just get the date
        output.append(
            f"Tag: {tag['name']:<20} Size: {size:.1f}MB  Last Updated: {last_updated}"
        )
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="List Docker image tags")
    parser.add_argument("image", help="Name of the Docker image (e.g., ubuntu, nginx)")
    args = parser.parse_args()

    tags = fetch_docker_tags(args.image)
    print(f"\nFound {len(tags)} tags for {args.image}:")
    print("-" * 70)
    print(format_tags(tags))


if __name__ == "__main__":
    main()
