// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

/* See avx2_trades.c for full permission, license (GPL-v3), and author (vaibhav) info. */
#ifndef AVX2_TRADES_H
#define AVX2_TRADES_H
#include <stdint.h>
#include <stddef.h>

int32_t avx2_best_ask_i32(const int32_t* prices, size_t count);
int32_t avx2_best_bid_i32(const int32_t* prices, size_t count);
void avx2_delta_f32(float* out, const float* prices, float base, size_t count);
#endif
