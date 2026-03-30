#!/usr/bin/env bash

set -euo pipefail

package_version="$1"
status=$(curl --silent --show-error --output /dev/null --write-out "%{http_code}" https://pypi.org/pypi/gingado/${package_version}/json)

if [[ "$status" == "200" ]]; then
  echo "Version $package_version is already published on PyPI. Aborting."
  exit 1
fi

if [[ "$status" != "404" ]]; then
  echo "Unexpected response from PyPI while checking version $package_version: HTTP $status. Aborting."
  exit 1
fi

echo "Version $package_version not yet on PyPI, proceeding."
