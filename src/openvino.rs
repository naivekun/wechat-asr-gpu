//! OpenVINO 声学前向。特征提取、窗口切分和 CTC 仍由 Rust 处理。
use crate::fbank::DIM;
use crate::weight_plan::WlPlan;
use anyhow::{bail, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

extern "C" {
    fn wasr_ov_create(
        blob: *const u8,
        plan: *const WlPlan,
        device: *const c_char,
        threads: c_int,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn wasr_ov_destroy(ctx: *mut c_void);
    fn wasr_ov_forward(
        ctx: *mut c_void,
        feats: *const f32,
        frames: usize,
        ids: *mut u32,
        error: *mut c_char,
        capacity: usize,
    ) -> bool;
    fn wasr_ov_warm(
        ctx: *mut c_void,
        frames: *const usize,
        count: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> bool;
}

pub struct OpenVino {
    ctx: *mut c_void,
    device: String,
}

// C++ owns the context and locks the single inference request during each call.
unsafe impl Send for OpenVino {}
unsafe impl Sync for OpenVino {}

fn message(error: &[c_char]) -> String {
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl OpenVino {
    pub fn new(blob: &[u8], plan: &WlPlan, device: &str) -> Result<Self> {
        let name = CString::new(device)?;
        let threads = std::env::var("WTL_THREADS")
            .ok()
            .and_then(|s| s.parse::<c_int>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(0);
        let mut error = [0; 2048];
        let ctx = unsafe {
            wasr_ov_create(
                blob.as_ptr(),
                plan,
                name.as_ptr(),
                threads,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ctx.is_null() {
            bail!("OpenVINO({device}) 初始化失败: {}", message(&error));
        }
        Ok(Self {
            ctx,
            device: device.into(),
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Compile or import the fixed-size requests in advance. CPU uses a dynamic
    /// model, so warming it is intentionally a no-op.
    pub fn warmup(&self, frames: &[usize]) -> Result<()> {
        if frames.is_empty() {
            return Ok(());
        }
        let mut error = [0; 2048];
        let ok = unsafe {
            wasr_ov_warm(
                self.ctx,
                frames.as_ptr(),
                frames.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if !ok {
            bail!("OpenVINO({}) warmup failed: {}", self.device, message(&error));
        }
        Ok(())
    }

    pub fn forward(&self, feats: &[[f32; DIM]]) -> Vec<u32> {
        let mut ids = vec![0; feats.len() / 5];
        if ids.is_empty() {
            return ids;
        }
        let mut error = [0; 2048];
        let ok = unsafe {
            wasr_ov_forward(
                self.ctx,
                feats.as_ptr().cast(),
                feats.len(),
                ids.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert!(
            ok,
            "OpenVINO({}) 推理失败: {}",
            self.device,
            message(&error)
        );
        ids
    }
}

impl Drop for OpenVino {
    fn drop(&mut self) {
        unsafe { wasr_ov_destroy(self.ctx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_request_across_frame_lengths() {
        // 共享零缓冲区容纳任一权重，输出 bias 使每个有效帧固定选择 token 1。
        let bias = 192 * 64 * 3 * 3 * size_of::<f32>();
        let mut blob = vec![0; bias + 2 * size_of::<f32>()];
        blob[bias + 4..].copy_from_slice(&1.0f32.to_ne_bytes());
        let plan = WlPlan {
            layers: std::ptr::null(),
            n_layers: 0,
            lin: 0,
            lin_s: 1.0,
            lin_b: 0,
            out: 0,
            out_s: 1.0,
            out_b: bias,
            vocab: 2,
            d: 1,
            ff: 1,
            flat: 192 * 20,
            heads: 1,
            c1: 0,
            c2: 0,
            c3: 0,
        };
        let model = OpenVino::new(&blob, &plan, "CPU").unwrap();
        model.warmup(&[5, 10]).unwrap();
        for frames in [0, 4, 5, 9, 10, 17, 5] {
            let input = vec![[0.0; DIM]; frames];
            assert_eq!(
                model.forward(&input),
                vec![1; frames / 5],
                "frames={frames}"
            );
        }
    }
}
