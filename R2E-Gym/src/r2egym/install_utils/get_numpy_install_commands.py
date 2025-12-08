import itertools
import os
import subprocess
import sys
import tempfile
import shutil
import concurrent.futures

NUMPY_REPO_URL = "https://github.com/numpy/numpy.git"

NUMPY_VERSIONS = [
    "v0.6.0",
    "v1.0",
    "v1.1.0",
    "v1.10.0",
    "v1.11.0",
    "v1.12.0",
    "v1.13.0",
    "v1.14.0",
    "v1.15.0",
    "v1.16.0",
    "v1.17.0",
    "v1.18.0",
    "v1.19.0",
    "v1.2.0",
    "v1.20.0",
    "v1.21.0",
    "v1.22.0",
    "v1.23.0",
    "v1.24.0",
    "v1.25.0",
    "v1.26.0",
    "v1.8.0",
    "v1.9.0",
    "v2.0.0",
    "v2.1.0",
    "v2.2.0",
]

BUILD_RESULTS_FILE = "build_results.txt"


def build_single_version(version_tag):
    base_temp_dir = tempfile.mkdtemp(prefix=f"numpy_{version_tag}_")
    logfile_path = os.path.join(base_temp_dir, f"build_commands_{version_tag}.log")

    try:
        # 1. Clone
        target_dir = os.path.join(base_temp_dir, "numpy_repo")
        cmd_clone = f"git clone {NUMPY_REPO_URL} {target_dir}"
        subprocess.run(
            cmd_clone,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 2. Checkout the specific tag
        cmd_checkout = f"cd {target_dir} && git checkout {version_tag}"
        subprocess.run(
            cmd_checkout,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 3. Submodule update
        cmd_submodule = f"cd {target_dir} && git submodule update --init --recursive"
        subprocess.run(
            cmd_submodule,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 4. cd into the repo
        os.chdir(target_dir)

        # 5. Build
        shutil.copyfile(
            "/home/gcpuser/r2e-edits-internal/src/r2e_edits/install_utils/numpy_install.sh",
            f"{target_dir}/numpy_install.sh",
        )

        cmd_build = f"bash numpy_install.sh"
        subprocess.run(
            cmd_build,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 6. Write the success to the results file
        with open(BUILD_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{version_tag} - SUCCESS\n")

        return True
    except Exception as e:
        # 6. Write the failure to the results file
        with open(BUILD_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{version_tag} - FAILURE\n")
        return False


def main():
    import concurrent.futures

    # Clean or create the overall results file
    if os.path.exists(BUILD_RESULTS_FILE):
        os.remove(BUILD_RESULTS_FILE)
    with open(BUILD_RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("# Build results for all numpy tags:\n")

    # We can run up to 60 builds in parallel
    max_workers = 30

    # Build all requested versions in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for version_tag in NUMPY_VERSIONS:
            # Submit a new process for each version
            fut = executor.submit(build_single_version, version_tag)
            future_map[fut] = version_tag

        # As each future completes, print the result
        for fut in concurrent.futures.as_completed(future_map):
            version_tag = future_map[fut]
            try:
                success = fut.result()
                if success:
                    print(
                        f"[RESULT] Numpy {version_tag} built successfully! (logged in {BUILD_RESULTS_FILE})"
                    )
                else:
                    print(
                        f"[RESULT] Numpy {version_tag} failed every combo. (see {BUILD_RESULTS_FILE})"
                    )
            except Exception as e:
                print(f"[ERROR] Numpy {version_tag} encountered an exception: {e}")

    print("All builds complete!")


if __name__ == "__main__":
    main()
