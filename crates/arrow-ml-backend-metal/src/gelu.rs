use arrow_ml_common::device_tensor::FFI_DeviceTensor;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::sync::OnceLock;

const THREADGROUP_SIZE: u64 = 256;

const GELU_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float SQRT_2_OVER_PI = 0.7978845608028654;
constant float GELU_COEFF = 0.044715;

kernel void gelu_f32(
    device const float *input  [[buffer(0)]],
    device float *output       [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    if (i >= count) return;

    float x = input[i];
    float inner = SQRT_2_OVER_PI * fma(GELU_COEFF, x * x * x, x);
    output[i] = 0.5 * x * (1.0 + tanh(inner));
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
        Self::grow_buf(device, &mut self.buf_output, &mut self.output_cap, output_bytes);
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
            .new_library_with_source(GELU_F32_MSL, &options)
            .map_err(|e| format!("MSL compile error: {e}"))?;
        let func = library
            .get_function("gelu_f32", None)
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

pub struct GeluKernel {
    ctx: &'static MetalContext,
    buffers: BufferCache,
    dtype: i32,
}

impl GeluKernel {
    pub fn new(dtype: i32) -> Result<Self, String> {
        let ctx = metal_context()?;
        Ok(GeluKernel {
            ctx,
            buffers: BufferCache::new(),
            dtype,
        })
    }

    pub unsafe fn invoke(
        &mut self,
        input: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
    ) -> Result<(), String> {
        if input.dtype != self.dtype || output.dtype != self.dtype {
            return Err(format!(
                "dtype mismatch: kernel {} but tensors {}/{}",
                self.dtype, input.dtype, output.dtype
            ));
        }
        if input.ndim != output.ndim {
            return Err(format!(
                "ndim mismatch: input {} vs output {}",
                input.ndim, output.ndim
            ));
        }

        let ndim = input.ndim as usize;
        let in_shape = unsafe { std::slice::from_raw_parts(input.shape, ndim) };
        let out_shape = unsafe { std::slice::from_raw_parts(output.shape, ndim) };
        if in_shape != out_shape {
            return Err("input and output shapes must match".to_string());
        }

        let total: usize = in_shape.iter().map(|&d| d as usize).product();
        let elem_size = std::mem::size_of::<f32>() as u64;
        let data_bytes = total as u64 * elem_size;

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

        let count_u32 = total as u32;

        let cmd_buf = self.ctx.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.ctx.pipeline);
        encoder.set_buffer(0, Some(buf_in), 0);
        encoder.set_buffer(1, Some(buf_out), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count_u32 as *const u32 as *const std::ffi::c_void,
        );

        let groups = MTLSize::new((total as u64).div_ceil(THREADGROUP_SIZE), 1, 1);
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
