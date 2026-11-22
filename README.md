# Hyper-Optimized AVX2 Trading Engine

- **Author / Final Authority:** vaibhav (holds final power over this project)
- **License:** GPL-v3 (default per user rules; never MIT unless user explicitly overrides)
- **Language:** Rust + C (explicit user permission granted; full reason in `src/main.rs`)
- **Machine:** x86_64 Intel i5-1335U (AVX2 + FMA), WSL2 Debian 13

## Build

```bash
# Note: source ~/ml_env/bin/activate is available if Python libs added later.
# This pure Rust+C project does not require it.
RUSTFLAGS='-C target-cpu=native -C opt-level=3' cargo build --release
```

## Web Info Used
- Rust `std::simd` docs (nightly-only portable SIMD): https://doc.rust-lang.org/std/simd/
- Rust `std::arch` AVX2 intrinsics & runtime detection (`is_x86_feature_detected!`): https://doc.rust-lang.org/std/arch/

## If Pushed Public
Ask vaibhav which license to apply. Default is GPL-v3 (already set).
