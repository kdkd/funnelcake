// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

//! Build script: links the in-tree core static library (libfunnelcake.a,
//! produced by `make lib`), which contains both the scaler API and the binding
//! helpers. No crates.io dependencies and nothing to compile here.

use std::path::PathBuf;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let root = PathBuf::from(&manifest)
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root");

    // Link the core static lib + libm. Order matters for GNU ld: libfunnelcake
    // references libm, so -lm follows.
    println!("cargo:rustc-link-search=native={}", root.display());
    println!("cargo:rustc-link-lib=static=funnelcake");
    println!("cargo:rustc-link-lib=dylib=m");

    println!("cargo:rerun-if-changed={}", root.join("libfunnelcake.a").display());
    println!("cargo:rerun-if-changed=build.rs");
}
