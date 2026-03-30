from pathlib import Path
import re


VERSION_FILE = Path("gingado/__init__.py")
VERSION_PATTERN = re.compile(r'^__version__ = ["\']([^"\']+)["\']', re.M)


def read_package_version() -> str:
    version_file = VERSION_FILE.read_text(encoding="utf-8")
    version_match = VERSION_PATTERN.search(version_file)
    if version_match is None:
        raise SystemExit("Unable to find package version in gingado/__init__.py")

    return version_match.group(1)


if __name__ == "__main__":
    print(read_package_version())
