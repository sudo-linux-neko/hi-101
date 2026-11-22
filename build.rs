// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

/* ============================================================================
 * Author: vaibhav (final authority over this project)
 * License: GPL-v3 (default, never MIT)
 * ============================================================================ */
use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let c_file = PathBuf::from("c_src/avx2_trades.c");

    cc::Build::new()
        .file(&c_file)
        .flag("-O3")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-std=c11")
        .compile("avx2_trades");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=avx2_trades");
}
