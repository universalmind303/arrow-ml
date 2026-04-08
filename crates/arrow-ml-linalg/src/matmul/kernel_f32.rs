use std::simd::prelude::*;
use std::simd::StdFloat;

use super::packing;

/// Wrapper to send a raw mutable pointer across threads.
/// SAFETY: caller must ensure disjoint access from each thread.
#[derive(Clone, Copy)]
struct RawMutPtr<T>(*mut T);
unsafe impl<T> Send for RawMutPtr<T> {}
unsafe impl<T> Sync for RawMutPtr<T> {}

impl<T> RawMutPtr<T> {
    fn as_ptr(self) -> *mut T {
        self.0
    }
}

// Micro-kernel dimensions
pub const MR: usize = 4;
pub const NR: usize = 16;
const SIMD_WIDTH: usize = 4;
const NR_VECS: usize = NR / SIMD_WIDTH; // 4

// Cache blocking parameters
const KC: usize = 256;
const MC: usize = 64;
const NC: usize = 4096;

/// 4x16 micro-kernel: computes a MR x NR block of C accumulated over `kc` depth.
///
/// `packed_a`: micro-panel of A, layout [kc][MR], length >= kc * MR
/// `packed_b`: micro-panel of B, layout [kc][NR], length >= kc * NR
/// `c`: pointer to the (ir, jr) corner of the output matrix, row-major with stride `ldc`
/// `first`: if true, overwrite C; if false, accumulate into C
#[inline(always)]
unsafe fn microkernel(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc: usize,
    mr_valid: usize,
    nr_valid: usize,
    first: bool,
) {
    // 4 rows x 4 SIMD vectors = 16 accumulator registers
    let mut acc = [[Simd::<f32, 4>::splat(0.0); NR_VECS]; MR];

    let a_ptr = packed_a.as_ptr();
    let b_ptr = packed_b.as_ptr();

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        // Load 4 SIMD vectors from packed B — no bounds checks
        let b0 = Simd::<f32, 4>::from_array(*(b_ptr.add(b_off) as *const [f32; 4]));
        let b1 = Simd::<f32, 4>::from_array(*(b_ptr.add(b_off + 4) as *const [f32; 4]));
        let b2 = Simd::<f32, 4>::from_array(*(b_ptr.add(b_off + 8) as *const [f32; 4]));
        let b3 = Simd::<f32, 4>::from_array(*(b_ptr.add(b_off + 12) as *const [f32; 4]));

        // Broadcast each A element and FMA into accumulators
        let a0 = Simd::splat(*a_ptr.add(a_off));
        acc[0][0] = a0.mul_add(b0, acc[0][0]);
        acc[0][1] = a0.mul_add(b1, acc[0][1]);
        acc[0][2] = a0.mul_add(b2, acc[0][2]);
        acc[0][3] = a0.mul_add(b3, acc[0][3]);

        let a1 = Simd::splat(*a_ptr.add(a_off + 1));
        acc[1][0] = a1.mul_add(b0, acc[1][0]);
        acc[1][1] = a1.mul_add(b1, acc[1][1]);
        acc[1][2] = a1.mul_add(b2, acc[1][2]);
        acc[1][3] = a1.mul_add(b3, acc[1][3]);

        let a2 = Simd::splat(*a_ptr.add(a_off + 2));
        acc[2][0] = a2.mul_add(b0, acc[2][0]);
        acc[2][1] = a2.mul_add(b1, acc[2][1]);
        acc[2][2] = a2.mul_add(b2, acc[2][2]);
        acc[2][3] = a2.mul_add(b3, acc[2][3]);

        let a3 = Simd::splat(*a_ptr.add(a_off + 3));
        acc[3][0] = a3.mul_add(b0, acc[3][0]);
        acc[3][1] = a3.mul_add(b1, acc[3][1]);
        acc[3][2] = a3.mul_add(b2, acc[3][2]);
        acc[3][3] = a3.mul_add(b3, acc[3][3]);
    }

    // Store results back to C
    if mr_valid == MR && nr_valid == NR {
        // Fast path: full tile — direct SIMD stores
        let c_ptr = c.as_mut_ptr();
        for i in 0..MR {
            let row_ptr = c_ptr.add(i * ldc);
            for jv in 0..NR_VECS {
                let dst = row_ptr.add(jv * SIMD_WIDTH);
                if first {
                    std::ptr::copy_nonoverlapping(acc[i][jv].as_array().as_ptr(), dst, SIMD_WIDTH);
                } else {
                    let existing = Simd::<f32, 4>::from_array(*(dst as *const [f32; 4]));
                    let result = existing + acc[i][jv];
                    std::ptr::copy_nonoverlapping(result.as_array().as_ptr(), dst, SIMD_WIDTH);
                }
            }
        }
    } else {
        // Edge case: partial tile — scalar store
        for i in 0..mr_valid {
            for jv in 0..NR_VECS {
                let arr = acc[i][jv].to_array();
                for jj in 0..SIMD_WIDTH {
                    let j = jv * SIMD_WIDTH + jj;
                    if j < nr_valid {
                        let idx = i * ldc + j;
                        if first {
                            *c.get_unchecked_mut(idx) = arr[jj];
                        } else {
                            *c.get_unchecked_mut(idx) += arr[jj];
                        }
                    }
                }
            }
        }
    }
}

/// Macro-kernel: processes one MC×NC tile of C, given packed A and shared packed B.
///
/// SAFETY: all packed buffer offsets and c_slice indexing must be in-bounds.
unsafe fn macrokernel(
    packed_a: &[f32],
    packed_b: &[f32],
    c_slice: &mut [f32],
    n: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    first: bool,
) {
    for ir in (0..mc).step_by(MR) {
        let mr_valid = MR.min(mc - ir);
        let a_panel_offset = (ir / MR) * MR * kc;

        for jr in (0..nc).step_by(NR) {
            let nr_valid = NR.min(nc - jr);
            let b_panel_offset = (jr / NR) * NR * kc;

            let c_offset = ir * n + jr;

            microkernel(
                &packed_a[a_panel_offset..],
                &packed_b[b_panel_offset..],
                &mut c_slice[c_offset..],
                n,
                kc,
                mr_valid,
                nr_valid,
                first,
            );
        }
    }
}

/// Tiled, packed, SIMD-vectorized, multi-threaded GEMM for f32.
///
/// Computes C = A * B where A is (m x k) and B is (k x n), all row-major.
#[allow(clippy::uninit_vec)]
pub fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let mut c = vec![0.0f32; m * n];

    // Pre-allocate packed B buffer (reused across iterations)
    let packed_b_len = KC * ((NC + NR - 1) / NR * NR);
    let mut packed_b = Vec::<f32>::with_capacity(packed_b_len);
    unsafe { packed_b.set_len(packed_b_len) };

    // Max packed A size per thread
    let packed_a_len = ((MC + MR - 1) / MR * MR) * KC;

    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);
            let first = pc == 0;

            // Pack B panel (shared read-only across all threads)
            packing::pack_b(b, n, pc, jc, kc, nc, NR, &mut packed_b);

            let ic_blocks: Vec<usize> = (0..m).step_by(MC).collect();

            // SAFETY: each thread writes to rows [ic..ic+mc], which are disjoint row bands.
            let c_ptr = RawMutPtr(c.as_mut_ptr());
            let c_len = c.len();

            // for_each_init: allocate packed_a once per worker thread, reuse across blocks
            ic_blocks.par_iter().for_each_init(
                || {
                    let mut buf = Vec::<f32>::with_capacity(packed_a_len);
                    unsafe { buf.set_len(packed_a_len) };
                    buf
                },
                |packed_a, &ic| {
                    let mc = MC.min(m - ic);
                    packing::pack_a(a, k, ic, pc, mc, kc, MR, packed_a);

                    unsafe {
                        let c_row_start = ic * n + jc;
                        let c_slice = std::slice::from_raw_parts_mut(
                            c_ptr.as_ptr().add(c_row_start),
                            c_len - c_row_start,
                        );
                        macrokernel(packed_a, &packed_b, c_slice, n, mc, nc, kc, first);
                    }
                },
            );
        }
    }

    c
}
