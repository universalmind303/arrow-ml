//! Metal GPU implementation of f32 matrix multiplication.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::sync::{Mutex, OnceLock};

/// Block dimensions matching the MSL kernel constants.
const BM: u64 = 64;
const BN: u64 = 64;
const TM: u64 = 4;
const TN: u64 = 4;
const THREADS_X: u64 = BM / TM; // 16
const THREADS_Y: u64 = BN / TN; // 16

/// MSL source for the high-performance tiled matmul kernel.
///
/// Each threadgroup computes a BM x BN (64 x 64) tile of C.
/// Each thread computes a TM x TN (4 x 4) sub-block.
/// Tiles of A and B are cooperatively loaded into threadgroup shared memory
/// with bank-conflict padding. Inner loop uses `fma()` for fused multiply-add.
const MATMUL_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;
constant uint TM = 4;
constant uint TN = 4;
constant uint THREADS_X = BM / TM;  // 16
constant uint THREADS_Y = BN / TN;  // 16
constant uint NUM_THREADS = THREADS_X * THREADS_Y;  // 256

kernel void matmul_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C       [[buffer(2)]],
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles with bank-conflict padding
    threadgroup float As[BM][BK + 1];
    threadgroup float Bs[BK][BN + 1];

    uint flat_id = tid.x * THREADS_Y + tid.y;

    // Base output position for this thread's TM x TN block
    uint c_row_base = bid.x * BM + tid.x * TM;
    uint c_col_base = bid.y * BN + tid.y * TN;

    // Accumulators
    float acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Iterate over K in BK-sized tiles
    for (uint bk = 0; bk < K; bk += BK) {
        // --- Cooperative load of A tile [BM x BK] ---
        for (uint idx = flat_id; idx < BM * BK; idx += NUM_THREADS) {
            uint m_local = idx / BK;
            uint k_local = idx % BK;
            uint m_global = bid.x * BM + m_local;
            uint k_global = bk + k_local;
            As[m_local][k_local] = (m_global < M && k_global < K)
                ? A[m_global * K + k_global] : 0.0f;
        }

        // --- Cooperative load of B tile [BK x BN] ---
        for (uint idx = flat_id; idx < BK * BN; idx += NUM_THREADS) {
            uint k_local = idx / BN;
            uint n_local = idx % BN;
            uint k_global = bk + k_local;
            uint n_global = bid.y * BN + n_local;
            Bs[k_local][n_local] = (k_global < K && n_global < N)
                ? B[k_global * N + n_global] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute TM x TN partial products from shared memory ---
        for (uint p = 0; p < BK; p++) {
            float a_reg[TM];
            for (uint tm = 0; tm < TM; tm++) {
                a_reg[tm] = As[tid.x * TM + tm][p];
            }
            for (uint tn = 0; tn < TN; tn++) {
                float b_val = Bs[p][tid.y * TN + tn];
                for (uint tm = 0; tm < TM; tm++) {
                    acc[tm][tn] = fma(a_reg[tm], b_val, acc[tm][tn]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Store TM x TN results to global memory ---
    for (uint tm = 0; tm < TM; tm++) {
        uint r = c_row_base + tm;
        if (r >= M) continue;
        for (uint tn = 0; tn < TN; tn++) {
            uint c = c_col_base + tn;
            if (c < N) {
                C[r * N + c] = acc[tm][tn];
            }
        }
    }
}
"#;

/// Cached Metal buffers that grow as needed and are reused across calls.
struct BufferCache {
    buf_a: Option<Buffer>,
    buf_b: Option<Buffer>,
    buf_c: Option<Buffer>,
    a_cap: u64,
    b_cap: u64,
    c_cap: u64,
}

impl BufferCache {
    fn new() -> Self {
        Self {
            buf_a: None,
            buf_b: None,
            buf_c: None,
            a_cap: 0,
            b_cap: 0,
            c_cap: 0,
        }
    }

    fn ensure_sizes(&mut self, device: &Device, a_bytes: u64, b_bytes: u64, c_bytes: u64) {
        Self::grow_buf(device, &mut self.buf_a, &mut self.a_cap, a_bytes);
        Self::grow_buf(device, &mut self.buf_b, &mut self.b_cap, b_bytes);
        Self::grow_buf(device, &mut self.buf_c, &mut self.c_cap, c_bytes);
    }

    fn grow_buf(device: &Device, cached: &mut Option<Buffer>, cap: &mut u64, byte_len: u64) {
        if *cap >= byte_len && cached.is_some() {
            return;
        }
        let alloc_size = byte_len.max((*cap * 3) / 2);
        let opts = MTLResourceOptions::StorageModeShared;
        let buf = device.new_buffer(alloc_size, opts);
        *cap = alloc_size;
        *cached = Some(buf);
    }
}

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    buffers: Mutex<BufferCache>,
}

// SAFETY: Metal's Device, CommandQueue, and ComputePipelineState are
// thread-safe Objective-C objects with internal synchronization.
// BufferCache is protected by a Mutex.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

fn get_context() -> Result<&'static MetalContext, String> {
    static CONTEXT: OnceLock<Result<MetalContext, String>> = OnceLock::new();

    let result = CONTEXT.get_or_init(|| {
        let device =
            Device::system_default().ok_or_else(|| "no Metal-capable GPU found".to_string())?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MATMUL_F32_MSL, &options)
            .map_err(|e| format!("MSL compile error: {e}"))?;
        let func = library
            .get_function("matmul_f32", None)
            .map_err(|e| format!("function lookup error: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline creation error: {e}"))?;
        Ok(MetalContext {
            device,
            queue,
            pipeline,
            buffers: Mutex::new(BufferCache::new()),
        })
    });

    match result {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(msg.clone()),
    }
}

/// Perform f32 matrix multiplication on the Metal GPU.
///
/// `a` is row-major (m x k), `b` is row-major (k x n).
/// Returns a `Vec<f32>` of length m*n containing the row-major result C = A*B.
pub fn metal_matmul_f32(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let ctx = get_context()?;

    let a_byte_len = (a.len() * std::mem::size_of::<f32>()) as u64;
    let b_byte_len = (b.len() * std::mem::size_of::<f32>()) as u64;
    let c_byte_len = (m * n * std::mem::size_of::<f32>()) as u64;

    let mut cache = ctx.buffers.lock().unwrap();
    cache.ensure_sizes(&ctx.device, a_byte_len, b_byte_len, c_byte_len);
    let buf_a = cache.buf_a.as_ref().unwrap();
    let buf_b = cache.buf_b.as_ref().unwrap();
    let buf_c = cache.buf_c.as_ref().unwrap();

    unsafe {
        std::ptr::copy_nonoverlapping(
            a.as_ptr() as *const u8,
            buf_a.contents() as *mut u8,
            a_byte_len as usize,
        );
        std::ptr::copy_nonoverlapping(
            b.as_ptr() as *const u8,
            buf_b.contents() as *mut u8,
            b_byte_len as usize,
        );
    }

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let cmd_buf = ctx.queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&ctx.pipeline);
    encoder.set_buffer(0, Some(buf_a), 0);
    encoder.set_buffer(1, Some(buf_b), 0);
    encoder.set_buffer(2, Some(buf_c), 0);

    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &m_u32 as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &n_u32 as *const u32 as *const std::ffi::c_void,
    );

    let groups = MTLSize::new((m as u64 + BM - 1) / BM, (n as u64 + BN - 1) / BN, 1);
    let threads_per_group = MTLSize::new(THREADS_X, THREADS_Y, 1);

    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let ptr = buf_c.contents() as *const f32;
    let result = unsafe { std::slice::from_raw_parts(ptr, m * n) }.to_vec();

    drop(cache);

    Ok(result)
}
