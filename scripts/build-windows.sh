#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
BUILD_DIR="${REPO_ROOT}/build/windows"
BUILD_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACTS=()

build_mingw=1
build_msvc=1

usage() {
  cat <<'EOF'
Usage: scripts/build-windows.sh [--all] [--mingw] [--msvc]

Build Windows release archives:
  --all     build every supported Windows target available on this host (default)
  --mingw   build MinGW-w64 static library packages
  --msvc    build an MSVC static library package from a Visual Studio shell

MinGW targets require cross compilers such as x86_64-w64-mingw32-gcc.
The MSVC target requires cl.exe and lib.exe in PATH, usually by running from
"x64 Native Tools Command Prompt for VS" or an equivalent Developer PowerShell.
EOF
}

if [ "$#" -gt 0 ]; then
  build_mingw=0
  build_msvc=0
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      build_mingw=1
      build_msvc=1
      ;;
    --mingw)
      build_mingw=1
      ;;
    --msvc)
      build_msvc=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

have_tool() {
  command -v "$1" >/dev/null 2>&1
}

hash_file() {
  local file="$1"

  if have_tool shasum; then
    shasum -a 256 "${file}"
  elif have_tool sha256sum; then
    sha256sum "${file}"
  else
    echo "error: shasum or sha256sum is required to write checksums" >&2
    exit 1
  fi
}

native_path() {
  if have_tool cygpath; then
    cygpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

make_package_dir() {
  local package_dir="$1"
  local target_arch="$2"
  local compiler="$3"
  local library_path="$4"
  local library_name="$5"

  rm -rf "${DIST_DIR}/${package_dir}"
  mkdir -p "${DIST_DIR}/${package_dir}/include"
  cp "${library_path}" "${DIST_DIR}/${package_dir}/${library_name}"
  cp "${REPO_ROOT}/include/funnelcake.h" "${DIST_DIR}/${package_dir}/include/"
  cp "${REPO_ROOT}/include/funnelcake_helpers.h" "${DIST_DIR}/${package_dir}/include/"
  cp "${REPO_ROOT}/README.md" "${REPO_ROOT}/INSTALL.md" "${DIST_DIR}/${package_dir}/"

  {
    printf '%s\n' "name=funnelcake"
    printf '%s\n' "target_os=windows"
    printf '%s\n' "base_distribution=${compiler}"
    printf '%s\n' "target_arch=${target_arch}"
    printf '%s\n' "compiler=${compiler}"
    printf '%s\n' "lto=0"
    printf '%s\n' "build_date=${BUILD_DATE}"
  } > "${DIST_DIR}/${package_dir}/BUILD_INFO"
}

make_zip() {
  local package_dir="$1"

  if ! have_tool zip; then
    echo "error: zip is required for Windows release archives" >&2
    exit 1
  fi

  rm -f "${DIST_DIR}/${package_dir}.zip" "${DIST_DIR}/${package_dir}.zip.sha256"
  (
    cd "${DIST_DIR}"
    zip -qr "${package_dir}.zip" "${package_dir}"
    hash_file "${package_dir}.zip" > "${package_dir}.zip.sha256"
  )
  ARTIFACTS+=("${package_dir}.zip" "${package_dir}.zip.sha256")
}

build_windows_mingw() {
  local arch="$1"
  local cc="$2"
  local ar="$3"
  local package_dir="funnelcake-windows-mingw-${arch}"

  if ! have_tool "${cc}" || ! have_tool "${ar}" || ! have_tool make; then
    echo "==> Skipping MinGW ${arch}: ${cc}, ${ar}, and make are required"
    return 0
  fi

  echo "==> Building Windows MinGW ${arch} artifact"
  (
    cd "${REPO_ROOT}"
    make clean
    make lib CC="${cc}" AR="${ar}" LTO=0 UNAME_S=Windows_NT UNAME_M="${arch}"
  )

  make_package_dir "${package_dir}" "${arch}" "mingw-w64" \
    "${REPO_ROOT}/libfunnelcake.a" "libfunnelcake.a"
  make_zip "${package_dir}"
}

msvc_arch_name() {
  case "${VSCMD_ARG_TGT_ARCH:-${PROCESSOR_ARCHITECTURE:-unknown}}" in
    x64|AMD64|amd64) printf '%s\n' "x64" ;;
    arm64|ARM64) printf '%s\n' "arm64" ;;
    *) printf '%s\n' "unknown" ;;
  esac
}

build_windows_msvc() {
  local arch
  local obj_dir
  local lib_path
  local package_dir
  local src
  local obj
  local objs=()
  local include_dir
  local src_dir
  local src_path
  local obj_path
  local lib_path_native
  local common_sources=(
    src/funnelcake.c
    src/funnelcake_hdr.c
    src/log.c
    src/detect.c
    src/kernels_scalar.c
    src/kernels_hdr_scalar.c
    src/tonemap.c
    src/kernels_upscale_scalar.c
    src/bindings_support.c
  )

  if ! have_tool cl || ! have_tool lib; then
    echo "==> Skipping MSVC: cl.exe and lib.exe are required"
    return 0
  fi

  arch="$(msvc_arch_name)"
  if [ "${arch}" = "unknown" ]; then
    echo "==> Skipping MSVC: unable to determine Visual Studio target architecture"
    return 0
  fi

  if [ "${arch}" = "arm64" ]; then
    common_sources+=(
      src/kernels_neon.c
      src/kernels_hdr_neon.c
      src/kernels_upscale_neon.c
      src/tonemap_neon.c
    )
  fi

  package_dir="funnelcake-windows-msvc-${arch}"
  obj_dir="${BUILD_DIR}/msvc-${arch}"
  lib_path="${obj_dir}/funnelcake.lib"
  include_dir="$(native_path "${REPO_ROOT}/include")"
  src_dir="$(native_path "${REPO_ROOT}/src")"
  lib_path_native="$(native_path "${lib_path}")"

  echo "==> Building Windows MSVC ${arch} artifact"
  rm -rf "${obj_dir}"
  mkdir -p "${obj_dir}"

  for src in "${common_sources[@]}"; do
    obj="${obj_dir}/$(basename "${src}" .c).obj"
    src_path="$(native_path "${REPO_ROOT}/${src}")"
    obj_path="$(native_path "${obj}")"
    MSYS2_ARG_CONV_EXCL='*' cl /nologo /std:c11 /O2 /W4 /WX \
      /D_CRT_SECURE_NO_WARNINGS /D_POSIX_C_SOURCE=200112L \
      /I"${include_dir}" /I"${src_dir}" \
      /Fo"${obj_path}" /c "${src_path}"
    objs+=("$(native_path "${obj}")")
  done

  MSYS2_ARG_CONV_EXCL='*' lib /nologo /OUT:"${lib_path_native}" "${objs[@]}"
  make_package_dir "${package_dir}" "${arch}" "msvc" "${lib_path}" "funnelcake.lib"
  make_zip "${package_dir}"
}

mkdir -p "${DIST_DIR}" "${BUILD_DIR}"

if [ "${build_mingw}" -eq 1 ]; then
  build_windows_mingw "x86_64" "x86_64-w64-mingw32-gcc" "x86_64-w64-mingw32-ar"
  build_windows_mingw "aarch64" "aarch64-w64-mingw32-gcc" "aarch64-w64-mingw32-ar"
fi

if [ "${build_msvc}" -eq 1 ]; then
  build_windows_msvc
fi

(
  cd "${REPO_ROOT}"
  make clean
)

if [ "${#ARTIFACTS[@]}" -eq 0 ]; then
  echo ""
  echo "No Windows artifacts were built."
  echo "Install MinGW-w64, or run --msvc from a Visual Studio developer shell."
  exit 1
fi

echo ""
echo "Artifacts written to ${DIST_DIR}:"
for artifact in "${ARTIFACTS[@]}"; do
  echo "  ${artifact}"
done
