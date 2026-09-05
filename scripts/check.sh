#!/bin/sh
# Copyright (c) 2020-2026 Kevin Day
# SPDX-License-Identifier: BSD-2-Clause-Patent
set -eu
make_cmd=${1:-make}
"$make_cmd" test check-native

if [ "${CHECK_BINDINGS:-1}" = 1 ]; then
    for pair in "${GO:-go}:test-go" "${CARGO:-cargo}:test-rust" "${PYTHON:-python3}:test-python"; do
        tool=${pair%:*}; target=${pair#*:}
        if command -v "$tool" >/dev/null 2>&1; then
            "$make_cmd" "$target"
        else
            echo "Skipping $target: $tool is unavailable"
        fi
    done
    major=$(${JAVAC:-javac} -version 2>&1 | sed -n 's/^javac \([0-9][0-9]*\).*/\1/p')
    if [ "${major:-0}" -ge 22 ]; then
        "$make_cmd" test-java
    else
        echo "Skipping test-java: JDK 22 or newer is unavailable"
    fi
fi

# Work in a disposable source tree so sanitizer and install checks do not
# replace the developer's normal objects or installed libraries.
check_dir=$(mktemp -d "${TMPDIR:-/tmp}/funnelcake-check.XXXXXX")
trap 'rm -rf "$check_dir"' EXIT HUP INT TERM
mkdir -p "$check_dir/source/src" "$check_dir/source/include" "$check_dir/source/test"
cp Makefile "$check_dir/source/"
cp include/*.h "$check_dir/source/include/"
cp src/*.c src/*.h src/*.inc "$check_dir/source/src/"
cp test/*.c test/*.h "$check_dir/source/test/"
cd "$check_dir/source"
"$make_cmd" clean > "$check_dir/clean.log" 2>&1

if [ "${CHECK_SANITIZERS:-1}" = 1 ]; then
    sanitizer_flags='-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all'
    "$make_cmd" test check-native LIB_OPT="$sanitizer_flags" TEST_OPT="$sanitizer_flags" LDFLAGS='-lm -fsanitize=address,undefined' SWSCALE_TEST_CFLAGS= SWSCALE_TEST_LDFLAGS=
fi

# Changing configuration back must replace sanitizer objects. Then another
# identical build must do no compilation or linking.
"$make_cmd" lib LIB_OPT=-O2 TEST_OPT=-O2 LTO=0 > "$check_dir/build.log" 2>&1
"$make_cmd" lib LIB_OPT=-O2 TEST_OPT=-O2 LTO=0 > "$check_dir/noop.log" 2>&1
if grep -E ' -c | rcs ' "$check_dir/noop.log"; then
    echo 'Identical configuration rebuilt objects unexpectedly' >&2
    exit 1
fi
"$make_cmd" funnelcake.pc PREFIX=/old-prefix
"$make_cmd" install PREFIX=/funnelcake-check DESTDIR="$check_dir/stage" LTO=0
pc="$check_dir/stage/funnelcake-check/lib/pkgconfig/funnelcake.pc"
test -f "$check_dir/stage/funnelcake-check/include/funnelcake_helpers.h"
grep -q '^prefix=/funnelcake-check$' "$pc"
if grep -q /old-prefix "$pc"; then
    echo 'Installed pkg-config metadata retained the old prefix' >&2
    exit 1
fi
cat > "$check_dir/consumer.c" <<'C'
#include <funnelcake.h>
#include <funnelcake_helpers.h>
int main(void) { return fused_version()[0] && fused_scaler_ctx_sizeof() ? 0 : 1; }
C
${CC:-cc} -I"$check_dir/stage/funnelcake-check/include" "$check_dir/consumer.c" "$check_dir/stage/funnelcake-check/lib/libfunnelcake.a" -lm -o "$check_dir/consumer"
"$check_dir/consumer"
echo 'All requested checks passed'
