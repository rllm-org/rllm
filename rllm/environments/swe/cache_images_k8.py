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


def create_daemonset_yaml(docker_image, name):
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name},
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

    try:
        # write the DaemonSet
        with open(yaml_file, "w") as f:
            yaml.safe_dump(create_daemonset_yaml(docker_image, ds_name), f)

        print(f"[apply] creating DaemonSet {ds_name} to pull {docker_image}")
        subprocess.run(["kubectl", "apply", "-f", yaml_file], check=True)

        # wait for it
        print(f"[wait] rollout status daemonset/{ds_name} (timeout 5m)")
        subprocess.run(
            ["kubectl", "rollout", "status", f"daemonset/{ds_name}", "--timeout=3600s"],
            check=True,
        )

        # clean up
        print(f"[delete] daemonset/{ds_name}")
        subprocess.run(["kubectl", "delete", "-f", yaml_file], check=True)
        os.remove(yaml_file)

        # update counter
        with counter_lock:
            processed_images.add(docker_image)
            total_images += 1
            current = total_images

        print(f"[ok] Cached {docker_image} ({current}/{total_targets})")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[error] kubectl failed: {e}")
        return False
    except Exception as e:
        print(f"[error] {e}")
        return False


# Load the dataset
dataset = load_dataset("R2E-Gym/R2E-Gym-V1", split="train")

# Extract unique docker images from the dataset
unique_images = set()
for entry in dataset:
    if "docker_image" in entry:
        unique_images.add(entry["docker_image"])

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
