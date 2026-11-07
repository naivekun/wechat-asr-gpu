#pragma once

#include <stddef.h>

typedef struct {
  size_t pn_g;
  size_t pn_b;
  size_t mn_g;
  size_t mn_b;
  // 字段顺序必须与 Rust 侧 src/weight_plan.rs 的 WlLayerPlan 逐字段一致（偏移会经过 padding 校验）
  size_t q;
  float qs;
  size_t k;
  float ks;
  size_t v;
  float vs;
  size_t o;
  float os;
  size_t f1;
  float f1s;
  size_t f1b;
  size_t f2;
  float f2s;
  size_t f2b;
} wl_layer_plan;

typedef struct {
  const wl_layer_plan* layers;
  int n_layers;
  size_t lin;
  float lin_s;
  size_t lin_b;
  size_t out;                        // 输出层 int8 权重
  float out_s;
  size_t out_b;                      // f32 bias
  int vocab;                         // 输出列数，MLX 含补齐列
  int d, ff, flat, heads;
  size_t c1, c2, c3;                 // 卷积权重 f32，xnet 布局 [O][I][KH][KW]
} wl_plan;
