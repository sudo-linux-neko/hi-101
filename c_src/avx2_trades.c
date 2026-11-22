// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

/* ============================================================================
 * Author / Final Authority: vaibhav
 * Project License: GNU General Public License v3.0 (GPL-v3) — see LICENSE file.
 * This file is part of the hyper-optimized trading-engine. No MIT; default GPL-v3.
 * ============================================================================
 * LANGUAGE MIX PERMISSION (FULL REASON DOCUMENTED):
 * The user explicitly selected "Rust + C". Pure Rust portable SIMD (std::simd)
 * is nightly-only and less optimized for this exact machine.
 * We use C with AVX2 intrinsics (immintrin.h) because:
 *  1. AVX2 256-bit registers process 8x i32 or 4x f64 per instruction.
 *  2. C allows direct inline assembly and tighter memory alignment control
 *     than Rust stable std::arch for this specific trading workload.
 *  3. Intel i5-1335U (AVX2-capable) achieves maximum throughput only with
 *     hand-tuned C SIMD loops operating on contiguous buffers.
 * This is the ONLY exception to pure-language rules and requires user consent.
 * ============================================================================
 * Machine target: x86_64, AVX2 enabled, WSL2 Debian 13 (trixie)
 * Build: must compile with -C target-cpu=native or -C target-feature=+avx2
 * ============================================================================ */

#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <stdbool.h>

/* Hyper-optimized AVX2 best-ask (minimum price) over int32 array. */
int32_t avx2_best_ask_i32(const int32_t* prices, size_t count) {
    if (count == 0) return 0x7FFFFFFF;
    if (count < 8) {
        int32_t best = prices[0];
        for (size_t i = 1; i < count; ++i)
            if (prices[i] < best) best = prices[i];
        return best;
    }

    __m256i best_vec = _mm256_set1_epi32(prices[0]);
    size_t i = 1;
    /* Process 8 elements per loop iteration using AVX2 256-bit registers. */
    for (; i + 8 <= count; i += 8) {
        __m256i load_vec = _mm256_load_si256((__m256i const*)(prices + i));
        best_vec = _mm256_min_epi32(best_vec, load_vec);
    }
    /* Horizontal reduction of the 8 int32 lanes. */
    int32_t best_buf[8];
    _mm256_storeu_si256((__m256i*)best_buf, best_vec);
    int32_t best = best_buf[0];
    for (int j = 1; j < 8; ++j)
        if (best_buf[j] < best) best = best_buf[j];
    /* Tail elements (non-multiple of 8). */
    for (; i < count; ++i) {
        if (prices[i] < best) best = prices[i];
    }
    return best;
}

/* Hyper-optimized AVX2 best-bid (maximum price) over int32 array. */
int32_t avx2_best_bid_i32(const int32_t* prices, size_t count) {
    if (count == 0) return 0;
    if (count < 8) {
        int32_t best = prices[0];
        for (size_t i = 1; i < count; ++i)
            if (prices[i] > best) best = prices[i];
        return best;
    }
    __m256i best_vec = _mm256_set1_epi32(prices[0]);
    size_t i = 1;
    for (; i + 8 <= count; i += 8) {
        __m256i load_vec = _mm256_load_si256((__m256i const*)(prices + i));
        best_vec = _mm256_max_epi32(best_vec, load_vec);
    }
    int32_t best_buf[8];
    _mm256_storeu_si256((__m256i*)best_buf, best_vec);
    int32_t best = best_buf[0];
    for (int j = 1; j < 8; ++j)
        if (best_buf[j] > best) best = best_buf[j];
    for (; i < count; ++i) {
        if (prices[i] > best) best = prices[i];
    }
    return best;
}

/* AVX2 batch price-delta calculation (prices - base) for 8 lanes at once. */
void avx2_delta_f32(float* out, const float* prices, float base, size_t count) {
    __m256 base_vec = _mm256_set1_ps(base);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 p_vec = _mm256_loadu_ps(prices + i);
        __m256 d_vec = _mm256_sub_ps(p_vec, base_vec);
        _mm256_storeu_ps(out + i, d_vec);
    }
    for (; i < count; ++i) out[i] = prices[i] - base;
}
