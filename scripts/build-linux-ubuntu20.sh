#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
BUILD_DATE="$(date -u +%Y%m%dT%H%M%SZ)"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required tool not found: $1" >&2
    exit 1
  fi
}

require_tool docker

if ! docker buildx version >/dev/null 2>&1; then
  echo "error: docker buildx is required" >&2
  exit 1
fi

mkdir -p "${DIST_DIR}"

build_arch() {
  local arch="$1"
  local platform
  local package_dir="funnelcake-linux-ubuntu20-${arch}"
  local export_dir="${DIST_DIR}/.${package_dir}.tmp"
  local output_root

  case "${arch}" in
    amd64) platform="linux/amd64" ;;
    arm64) platform="linux/arm64" ;;
    *)
      echo "error: unsupported architecture: ${arch}" >&2
      exit 1
      ;;
  esac

  rm -rf "${export_dir}"
  mkdir -p "${export_dir}"

  echo "==> Building ${arch} artifact in Ubuntu 20.04 (${platform})"
  docker buildx build \
    --platform "${platform}" \
    --build-arg TARGETARCH="${arch}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --output "type=local,dest=${export_dir}" \
    -f - "${REPO_ROOT}" <<'EOF'
FROM --platform=$TARGETPLATFORM ubuntu:20.04

ARG TARGETARCH
ARG BUILD_DATE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       clang \
       make \
       tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN make clean
RUN make lib CC=clang LTO=0

RUN pkg="/out/funnelcake-linux-ubuntu20-${TARGETARCH}" \
    && mkdir -p "${pkg}/include" \
    && cp libfunnelcake.a "${pkg}/" \
    && cp include/funnelcake.h include/funnelcake_helpers.h "${pkg}/include/" \
    && cp README.md INSTALL.md "${pkg}/" \
    && printf '%s\n' \
       "name=funnelcake" \
       "target_os=linux" \
       "base_distribution=ubuntu20.04" \
       "target_arch=${TARGETARCH}" \
       "compiler=clang" \
       "lto=0" \
       "build_date=${BUILD_DATE}" \
       > "${pkg}/BUILD_INFO"

WORKDIR /out
RUN pkg="funnelcake-linux-ubuntu20-${TARGETARCH}" \
    && tar -czf "${pkg}.tar.gz" "${pkg}" \
    && sha256sum "${pkg}.tar.gz" > "${pkg}.tar.gz.sha256"

FROM scratch
COPY --from=0 /out/ /out/
EOF

  output_root="${export_dir}/out"
  rm -rf "${DIST_DIR}/${package_dir}"
  rm -f "${DIST_DIR}/${package_dir}.tar.gz" "${DIST_DIR}/${package_dir}.tar.gz.sha256"
  mv "${output_root}/${package_dir}" "${DIST_DIR}/${package_dir}"
  mv "${output_root}/${package_dir}.tar.gz" "${DIST_DIR}/${package_dir}.tar.gz"
  mv "${output_root}/${package_dir}.tar.gz.sha256" "${DIST_DIR}/${package_dir}.tar.gz.sha256"
  rm -rf "${export_dir}"
}

build_arch amd64
build_arch arm64

echo ""
echo "Artifacts written to ${DIST_DIR}:"
echo "  funnelcake-linux-ubuntu20-amd64.tar.gz"
echo "  funnelcake-linux-ubuntu20-amd64.tar.gz.sha256"
echo "  funnelcake-linux-ubuntu20-arm64.tar.gz"
echo "  funnelcake-linux-ubuntu20-arm64.tar.gz.sha256"
