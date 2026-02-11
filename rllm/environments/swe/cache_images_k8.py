import concurrent.futures
import os
import subprocess
import threading
import uuid

import yaml
from datasets import load_dataset

counter_lock = threading.Lock()
total_images = 0
processed_images = set()

# Get namespace from environment variable, default to 'default'
KUBE_NAMESPACE = os.environ.get("KUBE_NAMESPACE", "default")
# Get Docker mirror prefix for 3rd party image source.
DOCKER_MIRROR_PREFIX = os.environ.get("DOCKER_MIRROR_PREFIX", "").strip()


def apply_mirror_prefix(docker_image: str) -> str:
    """Apply DOCKER_MIRROR_PREFIX to docker image name if configured."""
    if not DOCKER_MIRROR_PREFIX or not docker_image:
        return docker_image
    
    # Only add prefix if the image doesn't already have a registry
    # (i.e., doesn't contain a dot before the first slash, indicating no domain)
    if "/" in docker_image:
        first_part = docker_image.split("/")[0]
        # If first part doesn't contain a dot, it's not a registry domain
        if "." not in first_part:
            return f"{DOCKER_MIRROR_PREFIX}/{docker_image}"
        # Already has a registry domain, don't modify
        return docker_image
    else:
        # Image name without any slashes (e.g., "ubuntu:latest")
        return f"{DOCKER_MIRROR_PREFIX}/library/{docker_image}"


def create_daemonset_yaml(docker_image, name):
    # Apply mirror prefix to docker image
    docker_image = apply_mirror_prefix(docker_image)
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name, "namespace": KUBE_NAMESPACE},
        "spec": {
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "containers": [
                        {
                            "name": "image-puller",
                            "image": docker_image,
                            "command": ["sleep", "1000000"],
                            "imagePullPolicy": "Always",
                            "resources": {"requests": {"cpu": "0.5", "memory": "1Gi"}},
                        }
                    ],
                    "restartPolicy": "Always",
                    # "imagePullSecrets": [{"name": "dockerhub-pro"}],
                    # "nodeSelector": {"karpenter.sh/nodepool": "bigcpu-standby"},
                },
            },
        },
    }


def pull_image_on_all_nodes(docker_image, total_targets):
    global total_images, processed_images

    if docker_image in processed_images:
        print(f"[skip] {docker_image}")
        return True

    ds_name = f"image-puller-{uuid.uuid4().hex[:8]}"
    yaml_file = f"/tmp/{ds_name}.yaml"
    daemonset_created = False

    try:
        # write the DaemonSet
        with open(yaml_file, "w") as f:
            yaml.safe_dump(create_daemonset_yaml(docker_image, ds_name), f)

        prefixed_image = apply_mirror_prefix(docker_image)
        print(f"[apply] creating DaemonSet {ds_name} to pull {prefixed_image}")
        subprocess.run(["kubectl", "apply", "-f", yaml_file], check=True)
        daemonset_created = True

        # wait for it
        print(f"[wait] rollout status daemonset/{ds_name} (timeout 1h)")
        subprocess.run(
            ["kubectl", "rollout", "status", f"daemonset/{ds_name}", "--timeout=3600s"],
            check=True,
        )

        # update counter
        with counter_lock:
            processed_images.add(docker_image)
            total_images += 1
            current = total_images

        print(f"[ok] Cached {prefixed_image} ({current}/{total_targets})")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[error] kubectl failed for {docker_image}: {e}")
        return False
    except Exception as e:
        print(f"[error] {e}")
        return False
    finally:
        # ALWAYS clean up the DaemonSet and yaml file, even on failure
        if daemonset_created:
            try:
                print(f"[cleanup] deleting daemonset/{ds_name}")
                subprocess.run(
                    ["kubectl", "delete", "daemonset", ds_name, "-n", KUBE_NAMESPACE, "--ignore-not-found"],
                    check=False,  # Don't raise on error, we're cleaning up
                    timeout=60,
                )
            except Exception as cleanup_error:
                print(f"[cleanup-error] Failed to delete daemonset/{ds_name}: {cleanup_error}")
        
        # Clean up yaml file
        try:
            if os.path.exists(yaml_file):
                os.remove(yaml_file)
        except Exception:
            pass


# Load the dataset
dataset = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")

# Extract unique docker images from the dataset
unique_images = set()
for entry in dataset:
    if "docker_image" in entry:
        unique_images.add(entry["docker_image"])

print(f"Using Kubernetes namespace: {KUBE_NAMESPACE}")
print(f"Found {len(unique_images)} unique Docker images to cache")

# Process all unique images in parallel with 64 threads

# Process images using a ThreadPoolExecutor with 64 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
    # Submit all tasks to the executor
    future_to_image = {executor.submit(pull_image_on_all_nodes, image, len(unique_images)): image for image in unique_images}

    # Collect results as they complete
    results = []
    for future in concurrent.futures.as_completed(future_to_image):
        result = future.result()
        results.append(result)

print(f"Successfully cached {sum(results)} out of {len(unique_images)} Docker images on all nodes")
