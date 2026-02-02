#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_wikipedia_corpus(save_path: str):
    print("Downloading Wikipedia corpus from PeterJinGo/wiki-18-corpus...")

    wiki_dir = os.path.join(save_path, "wikipedia")
    os.makedirs(wiki_dir, exist_ok=True)

    wiki_file = os.path.join(wiki_dir, "wiki-18.jsonl")
    wiki_gz_file = os.path.join(wiki_dir, "wiki-18.jsonl.gz")

    if os.path.exists(wiki_file):
        print(f"Wikipedia corpus already exists at {wiki_file}")
        return wiki_file

    try:
        print("Downloading wiki-18.jsonl.gz from PeterJinGo/wiki-18-corpus...")

        hf_hub_download(
            repo_id="PeterJinGo/wiki-18-corpus",
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            local_dir=wiki_dir,
        )

        print(f"Downloaded compressed corpus to {wiki_gz_file}")

        print("Extracting tar archive from gzipped file...")

        try:
            with tarfile.open(wiki_gz_file, "r:gz") as tar:
                members = tar.getmembers()
                print(f"Found {len(members)} files in archive:")

                json_member = None
                for member in members:
                    print(f"  - {member.name} ({member.size} bytes)")
                    if member.name.endswith(".jsonl") or member.name.endswith(".json"):
                        json_member = member
                        break

                if json_member is None:
                    print("No .jsonl or .json file found in archive!")
                    return None

                print(f"Extracting {json_member.name}...")

                with tar.extractfile(json_member) as f_in:
                    with open(wiki_file, "wb") as f_out:
                        chunk_size = 8192
                        total_size = json_member.size
                        processed = 0

                        for chunk in tqdm(iter(lambda: f_in.read(chunk_size), b""), desc="Extracting corpus", total=total_size // chunk_size, unit="chunks"):
                            f_out.write(chunk)
                            processed += len(chunk)
                            if processed % (2000 * 1024 * 1024) == 0:  # print every 2GB
                                tqdm.write(f"Processed {processed / 1024 / 1024:.1f} MB")

        except tarfile.TarError as e:
            print(f"Failed to extract as tar archive: {e}")
            print("Falling back to binary extraction...")

            with gzip.open(wiki_gz_file, "rb") as f_in:
                with open(wiki_file, "wb") as f_out:
                    chunk_size = 8192
                    for chunk in tqdm(iter(lambda: f_in.read(chunk_size), b""), desc="Extracting corpus"):
                        f_out.write(chunk)

        print(f"Wikipedia corpus extracted to {wiki_file}")

        os.remove(wiki_gz_file)
        print("Removed compressed file to save space")

        return wiki_file

    except Exception as e:
        print(f"Error downloading Wikipedia corpus: {e}")
        print("Make sure you have internet connection and can access HuggingFace hub.")
        return None


def download_prebuilt_indices(save_path: str):
    print("Downloading pre-built E5 indices from PeterJinGo/wiki-18-e5-index...")

    indices_dir = os.path.join(save_path, "prebuilt_indices")
    os.makedirs(indices_dir, exist_ok=True)

    try:
        repo_id = "PeterJinGo/wiki-18-e5-index"
        for file in ["part_aa", "part_ab"]:
            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=indices_dir,
            )

        print(f"Pre-built indices downloaded to {indices_dir}")
        print("Note: You'll need to concatenate part_aa and part_ab to create the full index")
        print("Run: cat part_aa part_ab > e5_Flat.index")

        return indices_dir

    except Exception as e:
        print(f"Warning: Could not download pre-built indices: {e}")
        print("You can still run with sparse index only if needed")
        return None


def setup_search_data(data_dir: str = "./search_data", max_docs: int | None = None):
    print(f"Setting up Search data (dense-only) in {data_dir}")

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    wiki_file = download_wikipedia_corpus(data_dir)

    # Download dense index shards
    prebuilt_dir = download_prebuilt_indices(data_dir)

    summary = {
        "wikipedia_corpus": wiki_file,
        "prebuilt_dense_indices": prebuilt_dir,
        "setup_complete": wiki_file is not None and prebuilt_dir is not None,
    }

    summary_file = os.path.join(data_dir, "data_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nData setup {'completed' if summary['setup_complete'] else 'partially completed'}!")
    print(f"Summary saved to {summary_file}")

    if summary["setup_complete"]:
        print("\nNext steps:")
        print("1. Launch dense retrieval server:")
        print("   cd examples/search && bash retrieval/launch_server.sh ./search_data/prebuilt_indices 8000")
        print("2. Start training:")
        print("   cd examples/search && python train_search_agent.py")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download Search training data & dense indices")
    parser.add_argument("--data_dir", default="./search_data", help="Directory to save data")
    parser.add_argument("--max_docs", type=int, help="Limit number of docs (for quick dev)")

    args = parser.parse_args()

    setup_search_data(
        data_dir=args.data_dir,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
