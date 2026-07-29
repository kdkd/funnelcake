#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
BUILD_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
BUILD_DIR="${DIST_DIR}/build/macos-${HOST_ARCH}"

usage() {
  cat <<'EOF'
Usage: scripts/build-macos.sh

Build a native macOS release archive. This script must run on macOS with
Xcode command line tools installed.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 0 ]; then
  echo "error: unknown option: $1" >&2
  usage >&2
  exit 1
fi

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required tool not found: $1" >&2
    exit 1
  fi
}

if [ "${HOST_OS}" != "Darwin" ]; then
  echo "error: macOS artifacts require a Darwin host" >&2
  exit 1
fi

require_tool clang
require_tool make
require_tool shasum
require_tool tar

mkdir -p "${DIST_DIR}"

package_dir="funnelcake-macos-${HOST_ARCH}"

echo "==> Building macOS ${HOST_ARCH} artifact"
rm -rf "${BUILD_DIR}"
(
  cd "${REPO_ROOT}"
  make lib \
    BUILD_DIR="${BUILD_DIR}" \
    CC=clang \
    LTO=0 \
    UNAME_S=Darwin \
    UNAME_M="${HOST_ARCH}"
)

rm -rf "${DIST_DIR}/${package_dir}"
mkdir -p "${DIST_DIR}/${package_dir}/include"
cp "${BUILD_DIR}/libfunnelcake.a" "${DIST_DIR}/${package_dir}/"
cp "${REPO_ROOT}/include/funnelcake.h" "${DIST_DIR}/${package_dir}/include/"
cp "${REPO_ROOT}/include/funnelcake_helpers.h" "${DIST_DIR}/${package_dir}/include/"
cp "${REPO_ROOT}/README.md" "${REPO_ROOT}/INSTALL.md" "${DIST_DIR}/${package_dir}/"

{
  printf '%s\n' "name=funnelcake"
  printf '%s\n' "target_os=macos"
  printf '%s\n' "target_arch=${HOST_ARCH}"
  printf '%s\n' "compiler=clang"
  printf '%s\n' "lto=0"
  printf '%s\n' "build_date=${BUILD_DATE}"
} > "${DIST_DIR}/${package_dir}/BUILD_INFO"

rm -f "${DIST_DIR}/${package_dir}.tar.gz" "${DIST_DIR}/${package_dir}.tar.gz.sha256"
(
  cd "${DIST_DIR}"
  tar -czf "${package_dir}.tar.gz" "${package_dir}"
  shasum -a 256 "${package_dir}.tar.gz" > "${package_dir}.tar.gz.sha256"
)

echo ""
echo "Artifacts written to ${DIST_DIR}:"
echo "  build/macos-${HOST_ARCH}/"
echo "  ${package_dir}.tar.gz"
echo "  ${package_dir}.tar.gz.sha256"
