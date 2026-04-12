//! Metal GPU implementation of f32 softmax.
//!
//! Follows the same two-phase lifecycle as matmul: `SoftmaxKernel::new`
//! (called from `am_softmax_open`) and `SoftmaxKernel::invoke` (called
//! from `am_softmax_invoke`).
//!
//! The MSL kernel uses a threadgroup-cooperative parallel reduction:
//! one threadgroup per "row" to softmax (one (outer, inner) pair in the
//! outer/reduce/inner decomposition). Threads cooperate to find the max,
//! compute exp and sum, then normalize.

use arrow_ml_common::device_tensor::FFI_DeviceTensor;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::sync::OnceLock;

const THREADGROUP_SIZE: u64 = 256;

const SOFTMAX_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;

kernel void softmax_f32(
    device const float *input  [[buffer(0)]],
    device float *output       [[buffer(1)]],
    constant uint &outer_size  [[buffer(2)]],
    constant uint &dim_size    [[buffer(3)]],
    constant uint &inner_size  [[buffer(4)]],
    uint3 gid    [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tsize3 [[threads_per_threadgroup]]
) {
    threadgroup float shared[THREADGROUP_SIZE];

    uint outer_idx = gid.x;
    uint inner_idx = gid.y;
    uint tid = tid3.x;
    uint tcount = tsize3.x;
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    uint base = outer_idx * dim_size * inner_size + inner_idx;
    uint stride = inner_size;

    // Pass 1: find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint d = tid; d < dim_size; d += tcount) {
        local_max = max(local_max, input[base + d * stride]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0;
    for (uint d = tid; d < dim_size; d += tcount) {
        float val = exp(input[base + d * stride] - row_max);
        output[base + d * stride] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];

    // Pass 3: normalize
    for (uint d = tid; d < dim_size; d += tcount) {
        output[base + d * stride] /= row_sum;
    }
}
"#;

struct BufferCache {
    buf_input: Option<Buffer>,
    buf_output: Option<Buffer>,
    input_cap: u64,
    output_cap: u64,
}

impl BufferCache {
    fn new() -> Self {
        Self {
            buf_input: None,
            buf_output: None,
            input_cap: 0,
            output_cap: 0,
        }
    }

    fn ensure_sizes(&mut self, device: &Device, input_bytes: u64, output_bytes: u64) {
        Self::grow_buf(device, &mut self.buf_input, &mut self.input_cap, input_bytes);
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
            .new_library_with_source(SOFTMAX_F32_MSL, &options)
            .map_err(|e| format!("MSL compile error: {e}"))?;
        let func = library
            .get_function("softmax_f32", None)
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

pub struct SoftmaxKernel {
    ctx: &'static MetalContext,
    buffers: BufferCache,
    dtype: i32,
}

impl SoftmaxKernel {
    pub fn new(dtype: i32) -> Result<Self, String> {
        let ctx = metal_context()?;
        Ok(SoftmaxKernel {
            ctx,
            buffers: BufferCache::new(),
            dtype,
        })
    }

    /// # Safety
    ///
    /// `input` and `output` must be valid `FFI_DeviceTensor`s with non-null
    /// `shape`/`strides` and a non-null data buffer. They must have the same
    /// shape.
    pub unsafe fn invoke(
        &mut self,
        input: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
        axis: i32,
    ) -> Result<(), String> {
        if input.dtype != self.dtype || output.dtype != self.dtype {
            return Err(format!(
                "dtype mismatch: kernel {} but tensors {}/{}",
                self.dtype, input.dtype, output.dtype
            ));
        }
        let ndim = input.ndim as usize;
        if ndim == 0 {
            return Err("softmax requires at least 1D tensor".to_string());
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

        let outer_size: usize = in_shape[..axis_idx].iter().map(|&d| d as usize).product::<usize>().max(1);
        let dim_size = in_shape[axis_idx] as usize;
        let inner_size: usize = in_shape[axis_idx + 1..].iter().map(|&d| d as usize).product::<usize>().max(1);

        let total_elems = (outer_size * dim_size * inner_size) as u64;
        let elem_size = std::mem::size_of::<f32>() as u64;
        let data_bytes = total_elems * elem_size;

        self.buffers
            .ensure_sizes(&self.ctx.device, data_bytes, data_bytes);
        let buf_in = self.buffers.buf_input.as_ref().expect("ensured");
        let buf_out = self.buffers.buf_output.as_ref().expect("ensured");

        let in_ptr = input.buffer.data as *const u8;
        if in_ptr.is_null() {
            return Err("input tensor data buffer is null".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(in_ptr, buf_in.contents() as *mut u8, data_bytes as usize);
        }

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;

        let cmd_buf = self.ctx.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.ctx.pipeline);
        encoder.set_buffer(0, Some(buf_in), 0);
        encoder.set_buffer(1, Some(buf_out), 0);

        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &outer_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &dim_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &inner_u32 as *const u32 as *const std::ffi::c_void,
        );

        let groups = MTLSize::new(outer_size as u64, inner_size as u64, 1);
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
