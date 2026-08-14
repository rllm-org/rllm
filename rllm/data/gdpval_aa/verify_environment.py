"""Assert the built sandbox matches Artificial Analysis' published manifests.

Baked into the GDPval-AA image and run as the final build step, so a drifted
Debian snapshot or a silently-substituted wheel fails the build instead of
surfacing as an unexplained score difference weeks later.

Compares:
  * ``dpkg-query -W`` against the 762 pinned Debian packages.
  * the solver virtualenv's installed distributions against the 419 pinned
    Python packages (``pip`` and ``wheel`` bootstrap the venv and are exempt).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import metadata


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _read_manifest(path: str, separator: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, found, version = line.partition(separator)
            if not found:
                raise SystemExit(f"malformed pin in {path}: {line!r}")
            pins[name] = version
    return pins


def _installed_system_packages() -> dict[str, str]:
    output = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\\t${Version}\\t${Status}\\n"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    installed: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, version, status = parts
        # dpkg lists removed-but-not-purged packages too; only "install ok
        # installed" is actually present on the filesystem.
        if status.strip() == "install ok installed":
            installed[name] = version
    return installed


def _installed_python_packages() -> dict[str, str]:
    return {_normalize(dist.metadata["Name"]): dist.version for dist in metadata.distributions() if dist.metadata["Name"]}


def _compare(kind: str, expected: dict[str, str], actual: dict[str, str], *, exempt: frozenset[str]) -> list[str]:
    problems: list[str] = []
    for name, version in sorted(expected.items()):
        if name not in actual:
            problems.append(f"{kind}: MISSING {name}=={version}")
        elif actual[name] != version:
            problems.append(f"{kind}: VERSION MISMATCH {name}: want {version}, got {actual[name]}")
    for name in sorted(set(actual) - set(expected) - exempt):
        problems.append(f"{kind}: UNEXPECTED {name}=={actual[name]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-manifest", required=True)
    parser.add_argument("--python-manifest", required=True)
    parser.add_argument(
        "--python-exempt",
        default="pip,wheel",
        help="Comma-separated distributions allowed to be present without being pinned.",
    )
    args = parser.parse_args()

    problems: list[str] = []

    system_expected = _read_manifest(args.system_manifest, "=")
    problems += _compare("debian", system_expected, _installed_system_packages(), exempt=frozenset())

    python_expected = {_normalize(name): version for name, version in _read_manifest(args.python_manifest, "==").items()}
    exempt = frozenset(_normalize(name) for name in args.python_exempt.split(",") if name)
    problems += _compare("python", python_expected, _installed_python_packages(), exempt=exempt)

    if problems:
        print(f"GDPval-AA environment does not match the published manifest ({len(problems)} problem(s)):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"GDPval-AA environment verified: {len(system_expected)} Debian and {len(python_expected)} Python packages match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
