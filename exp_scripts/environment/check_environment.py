"""Read-only, CPU-only checks for the migrated experiment entry points."""
import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dependency-conflicts", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = root / "exp_scripts"
    failures = []
    for label, path in [("environment", root / ".venv/bin/python"),
                        ("checkpoint destination", scripts / "finqa-prpo-run/checkpoints")]:
        if not path.exists():
            failures.append(f"{label}: missing {path}")
        else:
            print(f"OK {label}: {path.resolve()}")
    for module in ["torch", "vllm", "vllm._C", "verl", "rllm", "peft",
                   "openai", "pyarrow", "pandas", "flash_attn"]:
        try:
            importlib.import_module(module)
            print(f"OK import {module}")
        except Exception as error:
            failures.append(f"import {module}: {type(error).__name__}")
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                             "--exclude-standard", "--", "exp_scripts"],
                            cwd=root, capture_output=True, check=True)
    count = 0
    for relative in set(listed.stdout.decode().split("\0")):
        path = root / relative
        if path.suffix not in {".sh", ".py"} or not path.is_file():
            continue
        count += 1
        if path.suffix == ".py":
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as error:
                failures.append(f"syntax {relative}: line {error.lineno}")
        elif subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode:
            failures.append(f"syntax {relative}")
    print(f"Checked {count} script files (no bytecode written)")
    for name in ["rllm", "verl", "vllm", "torch", "transformers", "numpy"]:
        try:
            print(f"VERSION {name}={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package {name}")
    uv = shutil.which("uv")
    if uv is None:
        failures.append("uv missing; dependency declarations not checked")
    else:
        checked = subprocess.run([uv, "--no-config", "pip", "check", "--python", sys.executable],
                                 capture_output=True, text=True)
        print(checked.stdout.strip())
        print(checked.stderr.strip())
        if checked.returncode:
            if args.allow_dependency_conflicts:
                print("WAIVED dependency conflicts; environment is NOT declaration-clean")
            else:
                failures.append("package dependency declarations conflict")
    for failure in failures:
        print(f"FAIL {failure}")
    print("FAILED" if failures else "Checks passed" +
          (" with explicit dependency waiver" if args.allow_dependency_conflicts else ""))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
