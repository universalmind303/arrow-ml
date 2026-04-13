//! Metal GPU implementation of f32 layer normalization.
//!
//! Same two-phase lifecycle as matmul/softmax. The MSL kernel uses
//! threadgroup-cooperative parallel reductions: one threadgroup per
//! "row" to normalize. Threads cooperate to compute mean, variance,
//! then normalize and apply scale (gamma) + shift (beta).

use arrow_ml_common::device_tensor::FFI_DeviceTensor;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::sync::OnceLock;

const THREADGROUP_SIZE: u64 = 256;

const LAYERNORM_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;

kernel void layernorm_f32(
    device const float *input  [[buffer(0)]],
    device const float *gamma  [[buffer(1)]],
    device const float *beta   [[buffer(2)]],
    device float *output       [[buffer(3)]],
    constant uint &outer_size  [[buffer(4)]],
    constant uint &dim_size    [[buffer(5)]],
    constant float &epsilon    [[buffer(6)]],
    uint3 gid    [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tsize3 [[threads_per_threadgroup]]
) {
    threadgroup float shared[THREADGROUP_SIZE];

    uint row = gid.x;
    uint tid = tid3.x;
    uint tcount = tsize3.x;
    if (row >= outer_size) return;

    uint base = row * dim_size;

    // Pass 1: compute sum for mean
    float local_sum = 0.0;
    for (uint d = tid; d < dim_size; d += tcount) {
        local_sum += input[base + d];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(dim_size);

    // Pass 2: compute sum of squared deviations for variance
    float local_var = 0.0;
    for (uint d = tid; d < dim_size; d += tcount) {
        float diff = input[base + d] - mean;
        local_var += diff * diff;
    }
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(dim_size) + epsilon);

    // Pass 3: normalize, scale, shift
    for (uint d = tid; d < dim_size; d += tcount) {
        float normalized = (input[base + d] - mean) * inv_std;
        output[base + d] = fma(normalized, gamma[d], beta[d]);
    }
}
"#;

struct BufferCache {
    buf_input: Option<Buffer>,
    buf_gamma: Option<Buffer>,
    buf_beta: Option<Buffer>,
    buf_output: Option<Buffer>,
    input_cap: u64,
    gamma_cap: u64,
    beta_cap: u64,
    output_cap: u64,
}

impl BufferCache {
    fn new() -> Self {
        Self {
            buf_input: None,
            buf_gamma: None,
            buf_beta: None,
            buf_output: None,
            input_cap: 0,
            gamma_cap: 0,
            beta_cap: 0,
            output_cap: 0,
        }
    }

    fn ensure_sizes(
        &mut self,
        device: &Device,
        input_bytes: u64,
        gamma_bytes: u64,
        beta_bytes: u64,
        output_bytes: u64,
    ) {
        Self::grow_buf(device, &mut self.buf_input, &mut self.input_cap, input_bytes);
        Self::grow_buf(device, &mut self.buf_gamma, &mut self.gamma_cap, gamma_bytes);
        Self::grow_buf(device, &mut self.buf_beta, &mut self.beta_cap, beta_bytes);
        Self::grow_buf(
            device,
            &mut self.buf_output,
            &mut self.output_cap,
            output_bytes,
        );
    }

    fn grow_buf(device: &Device, cached: &mut Option<Buffer>, cap: &mut u64, byte_len: u64) {
        if *cap >= byte_len && cached.is_some() {
            return;
        }
        let alloc_size = byte_len.max((*cap * 3) / 2);
        let buf = device.new_buffer(alloc_size, MTLResourceOptions::StorageModeShared);
        *cap = alloc_size;
        *cached = Some(buf);
    }
}

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

fn metal_context() -> Result<&'static MetalContext, String> {
    static CONTEXT: OnceLock<Result<MetalContext, String>> = OnceLock::new();

    let result = CONTEXT.get_or_init(|| {
        let device =
            Device::system_default().ok_or_else(|| "no Metal-capable GPU found".to_string())?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(LAYERNORM_F32_MSL, &options)
            .map_err(|e| format!("MSL compile error: {e}"))?;
        let func = library
            .get_function("layernorm_f32", None)
            .map_err(|e| format!("function lookup error: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline creation error: {e}"))?;
        Ok(MetalContext {
            device,
            queue,
            pipeline,
        })
    });

    match result {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(msg.clone()),
    }
}

pub struct LayerNormKernel {
    ctx: &'static MetalContext,
    buffers: BufferCache,
    dtype: i32,
}

impl LayerNormKernel {
    pub fn new(dtype: i32) -> Result<Self, String> {
        let ctx = metal_context()?;
        Ok(LayerNormKernel {
            ctx,
            buffers: BufferCache::new(),
            dtype,
        })
    }

    /// # Safety
    ///
    /// All tensor pointers must be valid. `input` and `output` must have
    /// the same shape. `gamma` and `beta` must be 1D with length equal to
    /// the size of the normalized axis.
    pub unsafe fn invoke(
        &mut self,
        input: &FFI_DeviceTensor,
        gamma: &FFI_DeviceTensor,
        beta: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
        axis: i32,
        epsilon: f32,
    ) -> Result<(), String> {
        if input.dtype != self.dtype
            || gamma.dtype != self.dtype
            || beta.dtype != self.dtype
            || output.dtype != self.dtype
        {
            return Err(format!(
                "dtype mismatch: kernel {} but tensors {}/{}/{}/{}",
                self.dtype, input.dtype, gamma.dtype, beta.dtype, output.dtype
            ));
        }

        let ndim = input.ndim as usize;
        if ndim == 0 {
            return Err("layernorm requires at least 1D tensor".to_string());
        }
        if input.ndim != output.ndim {
            return Err(format!(
                "ndim mismatch: input {} vs output {}",
                input.ndim, output.ndim
            ));
        }

        let in_shape = unsafe { std::slice::from_raw_parts(input.shape, ndim) };
        let out_shape = unsafe { std::slice::from_raw_parts(output.shape, ndim) };
        if in_shape != out_shape {
            return Err("input and output shapes must match".to_string());
        }

        let resolved_axis = if axis < 0 {
            ndim as i32 + axis
        } else {
            axis
        };
        if resolved_axis < 0 || resolved_axis >= ndim as i32 {
            return Err(format!(
                "axis {} out of range for {}D tensor",
                axis, ndim
            ));
        }
        let axis_idx = resolved_axis as usize;

        // Layer norm normalizes over axis_idx..end, so we flatten:
        // outer_size = product of dims before axis
        // dim_size = product of dims from axis onward (the normalized dimensions)
        let outer_size: usize = in_shape[..axis_idx]
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        let dim_size: usize = in_shape[axis_idx..]
            .iter()
            .map(|&d| d as usize)
            .product();

        if gamma.ndim != 1 || beta.ndim != 1 {
            return Err(format!(
                "gamma/beta must be 1D, got {}D/{}D",
                gamma.ndim, beta.ndim
            ));
        }
        let gamma_shape = unsafe { std::slice::from_raw_parts(gamma.shape, 1) };
        let beta_shape = unsafe { std::slice::from_raw_parts(beta.shape, 1) };
        if gamma_shape[0] as usize != dim_size || beta_shape[0] as usize != dim_size {
            return Err(format!(
                "gamma/beta length {} / {} doesn't match dim_size {}",
                gamma_shape[0], beta_shape[0], dim_size
            ));
        }

        let elem_size = std::mem::size_of::<f32>() as u64;
        let data_bytes = (outer_size * dim_size) as u64 * elem_size;
        let param_bytes = dim_size as u64 * elem_size;

        self.buffers.ensure_sizes(
            &self.ctx.device,
            data_bytes,
            param_bytes,
            param_bytes,
            data_bytes,
        );
        let buf_in = self.buffers.buf_input.as_ref().expect("ensured");
        let buf_gamma = self.buffers.buf_gamma.as_ref().expect("ensured");
        let buf_beta = self.buffers.buf_beta.as_ref().expect("ensured");
        let buf_out = self.buffers.buf_output.as_ref().expect("ensured");

        let in_ptr = input.buffer.data as *const u8;
        let gamma_ptr = gamma.buffer.data as *const u8;
        let beta_ptr = beta.buffer.data as *const u8;
        if in_ptr.is_null() || gamma_ptr.is_null() || beta_ptr.is_null() {
            return Err("null data pointer in input, gamma, or beta".to_string());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(in_ptr, buf_in.contents() as *mut u8, data_bytes as usize);
            std::ptr::copy_nonoverlapping(
                gamma_ptr,
                buf_gamma.contents() as *mut u8,
                param_bytes as usize,
            );
            std::ptr::copy_nonoverlapping(
                beta_ptr,
                buf_beta.contents() as *mut u8,
                param_bytes as usize,
            );
        }

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;

        let cmd_buf = self.ctx.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.ctx.pipeline);
        encoder.set_buffer(0, Some(buf_in), 0);
        encoder.set_buffer(1, Some(buf_gamma), 0);
        encoder.set_buffer(2, Some(buf_beta), 0);
        encoder.set_buffer(3, Some(buf_out), 0);

        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &outer_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &dim_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<f32>() as u64,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );

        let groups = MTLSize::new(outer_size as u64, 1, 1);
        let threads_per_group = MTLSize::new(THREADGROUP_SIZE, 1, 1);

        encoder.dispatch_thread_groups(groups, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let out_ptr = output.buffer.data as *mut u8;
        if out_ptr.is_null() {
            return Err("output tensor data buffer is null".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_out.contents() as *const u8,
                out_ptr,
                data_bytes as usize,
            );
        }

        Ok(())
    }
}
