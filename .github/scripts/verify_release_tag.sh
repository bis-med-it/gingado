#!/usr/bin/env bash

set -euo pipefail

package_version="$1"
tag_version="${GITHUB_REF_NAME#v}"

if [[ "$package_version" != "$tag_version" ]]; then
  echo "Tag $GITHUB_REF_NAME does not match package version $package_version. Aborting."
  exit 1
fi

if ! git branch -r --contains "$GITHUB_SHA" | grep -q 'origin/main'; then
  echo "Tagged commit $GITHUB_SHA is not reachable from origin/main. Aborting."
  exit 1
fi

echo "Version check passed: $package_version"
