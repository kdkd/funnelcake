#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
SOURCE_DIR="${TEST_ROOT}/source"
BUILD_DIR="${TEST_ROOT}/build"

if [ "$(uname -s)" = "Darwin" ]; then
  shared_library="libfunnelcake.1.dylib"
else
  shared_library="libfunnelcake.so.1"
fi

cleanup() {
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT

mkdir -p "${SOURCE_DIR}"
cp "${REPO_ROOT}/Makefile" "${SOURCE_DIR}/"
cp -R "${REPO_ROOT}/include" "${REPO_ROOT}/src" "${SOURCE_DIR}/"

(
  cd "${SOURCE_DIR}"
  make clean
)

(
  cd "${SOURCE_DIR}"
  make lib shared BUILD_DIR="${BUILD_DIR}" LTO=0
)

if [ ! -f "${BUILD_DIR}/libfunnelcake.a" ]; then
  echo "error: static library was not written to BUILD_DIR" >&2
  exit 1
fi

if [ ! -f "${BUILD_DIR}/${shared_library}" ]; then
  echo "error: shared library was not written to BUILD_DIR" >&2
  exit 1
fi

if [ -f "${SOURCE_DIR}/libfunnelcake.a" ]; then
  echo "error: static library polluted the source directory" >&2
  exit 1
fi

if find "${SOURCE_DIR}/src" -name '*.o' -print -quit | grep -q .; then
  echo "error: object files polluted the source directory" >&2
  exit 1
fi

(
  cd "${SOURCE_DIR}"
  make clean BUILD_DIR="${BUILD_DIR}"
)

if find "${BUILD_DIR}" -type f -print -quit | grep -q .; then
  echo "error: clean left library build artifacts in BUILD_DIR" >&2
  exit 1
fi
