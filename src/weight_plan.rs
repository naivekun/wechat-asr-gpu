//! Rust 与 MLX / OpenVINO 共享的权重偏移表 ABI。

/// 一层的权重位置（blob 内字节偏移）+ per-tensor 量化 scale。
/// 字段顺序必须与 csrc_common/weight_plan.h 的 wl_layer_plan 完全一致。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WlLayerPlan {
    pub pn_g: usize,
    pub pn_b: usize,
    pub mn_g: usize,
    pub mn_b: usize,
    pub q: usize,
    pub qs: f32,
    pub k: usize,
    pub ks: f32,
    pub v: usize,
    pub vs: f32,
    pub o: usize,
    pub os: f32,
    pub f1: usize,
    pub f1s: f32,
    pub f1b: usize,
    pub f2: usize,
    pub f2s: f32,
    pub f2b: usize,
}

/// 整个模型的权重位置表（对应 wl_plan）。
#[repr(C)]
pub struct WlPlan {
    pub layers: *const WlLayerPlan,
    pub n_layers: i32,
    pub lin: usize,
    pub lin_s: f32,
    pub lin_b: usize,
    pub out: usize,
    pub out_s: f32,
    pub out_b: usize,
    pub vocab: i32,
    pub d: i32,
    pub ff: i32,
    pub flat: i32,
    pub heads: i32,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
}
