//! Indexed mixture-of-experts forward: one quantized mat-vec per routed pair.
//!
//! The weights are a single `(num_experts, n, k)` quantized tensor; `ids` picks
//! which expert each of the `batch * topk` routed tokens goes to. The kernel is
//! `indexed_moe_forward_*_q8_1` in `candle-kernels/src/quantized.cu` — the same
//! `vec_dot_q*_q8_1` inner loop [`super::mmvq`] uses, with the expert index
//! folded into the weight pointer, so it inherits MMVQ's `q8_1` activation
//! requantization wholesale (see [`super::q8_1`]).
//!
//! Mirrors `quantized/cuda.rs::indexed_moe_forward_fused_q8_1_input`.

use super::kernels::{arg, launch_err, MATRIX_ROW_PADDING, WARP_SIZE};
use super::q8_1::{buffer_bytes, pad, quantize_q8_1};
use super::QRocmStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{kernels, RocmStorage, RocmStorageSlice};
use crate::{Layout, Result, Shape};

/// `qi` (the MMVQ `QI*` constant) and `vdr` (`VDR_*_Q8_1_MMVQ`) for `dtype`,
/// taken verbatim from `quantized.cu`. Together with `dtype.block_size()`
/// (`qk`) these determine `blocks_per_iter = vdr * nwarps * WARP_SIZE / qi`,
/// the per-iteration reduction width `indexed_moe_forward` divides `k` by.
///
/// Mirrors `quantized/cuda.rs::moe_qi_vdr`.
fn qi_vdr(dtype: GgmlDType) -> Option<(usize, usize)> {
    let pair = match dtype {
        GgmlDType::Q2K => (16, 1),
        GgmlDType::Q3K => (16, 1),
        GgmlDType::Q4K => (32, 2),
        GgmlDType::Q5K => (32, 2),
        GgmlDType::Q6K => (32, 1),
        GgmlDType::Q8_0 => (8, 2),
        _ => return None,
    };
    Some(pair)
}

/// Kernel entry point and `nwarps` for `dtype` at reduction width `k`, or
/// `None` when there is no kernel for `dtype`.
///
/// `nwarps` is a compile-time template parameter of `indexed_moe_forward`
/// (it sizes the `tmp_shared[nwarps - 1][WARP_SIZE]` inter-warp reduction
/// buffer), so `blockDim.y` has to match whichever entry point this returns.
///
/// The kernel's reduction loop walks `k / qk` blocks in strides of
/// `blocks_per_iter = vdr * nwarps * WARP_SIZE / qi`. At `nwarps = 4` that
/// stride can exceed the number of blocks `k` actually has. For example,
/// `k = 768` against `Q4K` (`qk = 256`, `qi = 32`, `vdr = 2`) has 3 blocks
/// per row but a stride of 8, so more than half the launched warps find
/// `kbx >= blocks_per_row_x` on their very first (only) look and never do a
/// single `vec_dot`. The fallback is `nwarps = 1`, the smallest the kernel
/// supports. It keeps every warp's stride at `blocks_per_row_x` or below
/// whenever 4 warps would have been overkill for `k`.
///
/// Note the casing: the kernel names spell the K-quants `q4k`, not `q4_K` as
/// the MMVQ family does. Taken verbatim from `quantized.cu`.
///
/// Mirrors `quantized/cuda.rs::moe_kernel_for`.
fn kernel_for(dtype: GgmlDType, k: usize) -> Option<(&'static str, usize)> {
    let (name_nw4, name_nw1) = match dtype {
        GgmlDType::Q2K => (
            "indexed_moe_forward_q2k_q8_1",
            "indexed_moe_forward_q2k_q8_1_nw1",
        ),
        GgmlDType::Q3K => (
            "indexed_moe_forward_q3k_q8_1",
            "indexed_moe_forward_q3k_q8_1_nw1",
        ),
        GgmlDType::Q4K => (
            "indexed_moe_forward_q4k_q8_1",
            "indexed_moe_forward_q4k_q8_1_nw1",
        ),
        GgmlDType::Q5K => (
            "indexed_moe_forward_q5k_q8_1",
            "indexed_moe_forward_q5k_q8_1_nw1",
        ),
        GgmlDType::Q6K => (
            "indexed_moe_forward_q6k_q8_1",
            "indexed_moe_forward_q6k_q8_1_nw1",
        ),
        GgmlDType::Q8_0 => (
            "indexed_moe_forward_q8_0_q8_1",
            "indexed_moe_forward_q8_0_q8_1_nw1",
        ),
        _ => return None,
    };
    let (qi, vdr) = qi_vdr(dtype)?;
    let qk = dtype.block_size();
    let blocks_per_row = k / qk;
    // The largest nwarps (out of {1, 4}, the two instantiated tiers) whose
    // `blocks_per_iter` does not exceed `blocks_per_row`.
    let nwarps4_fits = vdr * 4 * WARP_SIZE / qi <= blocks_per_row;
    if nwarps4_fits {
        Some((name_nw4, 4))
    } else {
        Some((name_nw1, 1))
    }
}

/// The shapes the kernel launch is derived from, once validated.
struct Dims {
    num_experts: usize,
    n: usize,
    k: usize,
    batch: usize,
    /// `1` when every expert of a token shares one activation row, `topk` when
    /// each routed pair carries its own.
    input_dim1: usize,
    topk: usize,
}

fn dims(self_shape: &Shape, input_l: &Layout, ids_l: &Layout) -> Result<Dims> {
    let (num_experts, n, k) = self_shape.dims3()?;
    let (batch, input_dim1, input_k) = input_l.shape().dims3()?;
    let (ids_batch, topk) = ids_l.shape().dims2()?;
    if input_k != k {
        crate::bail!(
            "indexed_moe_forward: weights are {self_shape:?} but the input has k={input_k}"
        )
    }
    if ids_batch != batch {
        crate::bail!("indexed_moe_forward: input batch {batch} but ids batch {ids_batch}")
    }
    if input_dim1 != 1 && input_dim1 != topk {
        crate::bail!("indexed_moe_forward: input dim 1 is {input_dim1}, expected 1 or topk {topk}")
    }
    if topk == 0 || batch == 0 || n == 0 || k == 0 {
        crate::bail!(
            "indexed_moe_forward: empty shape {self_shape:?} / {:?}",
            ids_l.shape()
        )
    }
    Ok(Dims {
        num_experts,
        n,
        k,
        batch,
        input_dim1,
        topk,
    })
}

/// `q` is `(num_experts, n, k)`, `input` is `(batch, topk or 1, k)` f32 and
/// `ids` is `(batch, topk)` u32. Returns `(batch, topk, n)` f32.
pub(super) fn forward(
    q: &QRocmStorage,
    self_shape: &Shape,
    input: &RocmStorage,
    input_l: &Layout,
    ids: &RocmStorage,
    ids_l: &Layout,
) -> Result<(RocmStorage, Shape)> {
    let d = dims(self_shape, input_l, ids_l)?;
    if !d.k.is_multiple_of(q.dtype.block_size()) {
        crate::bail!(
            "indexed_moe_forward: k={} is not a multiple of the {:?} block size {}",
            d.k,
            q.dtype,
            q.dtype.block_size()
        )
    }
    // The kernel strides between experts by `n * k / block_size` blocks with no
    // bound of its own, so a short payload would read past the allocation.
    let data_elems = q.len / q.dtype.type_size() * q.dtype.block_size();
    if data_elems < d.num_experts * d.n * d.k {
        crate::bail!(
            "indexed_moe_forward: weights hold {data_elems} elems, need {}",
            d.num_experts * d.n * d.k
        )
    }

    let (y, y_offset) = match (&input.slice, input_l.contiguous_offsets()) {
        (RocmStorageSlice::F32(y), Some((o1, o2))) if o2 - o1 == d.batch * d.input_dim1 * d.k => {
            (y, o1)
        }
        (RocmStorageSlice::F32(_), _) => {
            crate::bail!("indexed_moe_forward expects a contiguous input, got {input_l:?}")
        }
        (slice, _) => crate::bail!(
            "indexed_moe_forward expects an f32 input, got {:?}",
            slice.dtype()
        ),
    };
    let (ids_mem, ids_offset) = match (&ids.slice, ids_l.contiguous_offsets()) {
        (RocmStorageSlice::U32(m), Some((o1, o2))) if o2 - o1 == d.batch * d.topk => (m, o1),
        (RocmStorageSlice::U32(_), _) => {
            crate::bail!("indexed_moe_forward expects contiguous u32 ids, got {ids_l:?}")
        }
        (slice, _) => crate::bail!(
            "indexed_moe_forward expects u32 ids, got {:?}",
            slice.dtype()
        ),
    };

    let (name, nwarps) = match kernel_for(q.dtype, d.k) {
        Some(plan) => plan,
        None => crate::bail!(
            "indexed_moe_forward is not implemented for {:?} on ROCm; \
             it needs one of q2k, q3k, q4k, q5k, q6k or q8_0",
            q.dtype
        ),
    };

    let dev = &q.device;
    let total_rows = d.batch * d.input_dim1;
    let k_padded = pad(d.k, MATRIX_ROW_PADDING);
    let input_q8_1 = dev.alloc_zeros::<u8>(buffer_bytes(d.k, total_rows))?;
    quantize_q8_1(y, y_offset, &input_q8_1, d.k, total_rows, dev)?;

    let out = dev.alloc_zeros::<f32>(d.batch * d.topk * d.n)?;
    let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;

    let w_ptr = q.data.as_ptr();
    let y_ptr = input_q8_1.as_ptr();
    // SAFETY: `ids_offset` is within the buffer — `contiguous_offsets` returned
    // it against this layout and the u32 storage backing it.
    let ids_ptr = unsafe { ids_mem.ptr_at(ids_offset) };
    let out_ptr = out.as_ptr();
    let n_i = d.n as i32;
    let k_i = d.k as i32;
    let batch_i = d.batch as i32;
    let topk_i = d.topk as i32;
    let k_padded_i = k_padded as i32;
    let input_dim1_i = d.input_dim1 as i32;
    let mut args = vec![
        arg(&w_ptr),
        arg(&y_ptr),
        arg(&ids_ptr),
        arg(&out_ptr),
        arg(&n_i),
        arg(&k_i),
        arg(&batch_i),
        arg(&topk_i),
        arg(&k_padded_i),
        arg(&input_dim1_i),
    ];
    // One block per (output row, batch, routed expert): `blockIdx.x` is the row
    // and the kernel flattens `(blockIdx.y, blockIdx.z)` into the task id it
    // indexes `ids` with.
    func.launch(
        Dim3::new_3d(d.n as u32, d.batch as u32, d.topk as u32),
        Dim3::new_2d(WARP_SIZE as u32, nwarps as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(name, e))?;

    Ok((
        RocmStorage {
            slice: RocmStorageSlice::F32(out),
            device: dev.clone(),
        },
        (d.batch, d.topk, d.n).into(),
    ))
}

#[cfg(test)]
mod tests;
