#![cfg(target_os = "macos")]

use arrow::datatypes::DataType;
use arrow_ml_common::device_tensor::{dtype, AmDeviceType};
use arrow_ml_common::kernels::softmax::SoftmaxKernel;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use std::mem;

fn require_metal_softmax() -> std::sync::Arc<arrow_ml_common::backend::Backend> {
    let reg = BackendRegistry::global();
    reg.best_softmax_for(dtype::FLOAT32, AmDeviceType::Metal as i32)
        .expect("Metal backend with softmax support not loaded")
}

fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let elem = mem::size_of::<f32>();
    let ndim = shape.len();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = elem;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    let buf = DeviceBuffer::from(arrow::buffer::Buffer::from_slice_ref(&data));
    Tensor::new(DataType::Float32, buf, Some(shape), Some(strides))
}

fn run_softmax(input: &Tensor, axis: i32) -> Tensor {
    let backend = require_metal_softmax();
    let kernel = SoftmaxKernel::open(backend, dtype::FLOAT32, AmDeviceType::Metal as i32).unwrap();

    let shape = input.shape().unwrap();
    let total: usize = shape.iter().product();
    let elem = mem::size_of::<f32>();
    let c_bytes = total * elem;

    let ndim = shape.len();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = elem;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    let output = Tensor::new(
        DataType::Float32,
        DeviceBuffer::new(c_bytes, Device::metal(0)),
        Some(shape.to_vec()),
        Some(strides),
    );

    let in_metal = input.to(Device::metal(0));
    let in_ffi = in_metal.as_ffi();
    let mut out_ffi = output.as_ffi();

    unsafe {
        kernel.invoke(&in_ffi, &mut out_ffi, axis).unwrap();
    }

    output
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    t.to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn softmax_1d_sums_to_one() {
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let output = run_softmax(&input, 0);
    let data = read_f32(&output);

    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "expected sum ~1.0, got {sum}");
}

#[test]
fn softmax_1d_ordering() {
    let input = make_tensor(vec![1.0, 3.0, 2.0], vec![3]);
    let output = run_softmax(&input, 0);
    let data = read_f32(&output);

    assert!(data[1] > data[2], "3.0 should have higher prob than 2.0");
    assert!(data[2] > data[0], "2.0 should have higher prob than 1.0");
}

#[test]
fn softmax_1d_uniform() {
    let input = make_tensor(vec![5.0, 5.0, 5.0, 5.0], vec![4]);
    let output = run_softmax(&input, 0);
    let data = read_f32(&output);

    for v in &data {
        assert!(
            (v - 0.25).abs() < 1e-5,
            "expected 0.25 for uniform input, got {v}"
        );
    }
}

#[test]
fn softmax_1d_numerical_stability() {
    let input = make_tensor(vec![1000.0, 1001.0, 1002.0], vec![3]);
    let output = run_softmax(&input, 0);
    let data = read_f32(&output);

    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "large values should still sum to 1, got {sum}"
    );
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);
}

#[test]
fn softmax_2d_last_axis() {
    // 2x3 tensor, softmax over axis 1 (rows)
    let input = make_tensor(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let output = run_softmax(&input, 1);
    let data = read_f32(&output);

    let row0_sum: f32 = data[0..3].iter().sum();
    let row1_sum: f32 = data[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum: {row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum: {row1_sum}");
}

#[test]
fn softmax_2d_first_axis() {
    // 2x3 tensor, softmax over axis 0 (columns)
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let output = run_softmax(&input, 0);
    let data = read_f32(&output);

    for j in 0..3 {
        let col_sum = data[j] + data[3 + j];
        assert!((col_sum - 1.0).abs() < 1e-5, "column {j} sum: {col_sum}");
    }
}

#[test]
fn softmax_2d_negative_axis() {
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let output = run_softmax(&input, -1); // same as axis=1
    let data = read_f32(&output);

    let row0_sum: f32 = data[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum: {row0_sum}");
}

#[test]
fn softmax_3d_attention_pattern() {
    // (batch=1, heads=2, seq_len=3), softmax over last axis
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let output = run_softmax(&input, -1);
    let data = read_f32(&output);

    let sum0: f32 = data[0..3].iter().sum();
    let sum1: f32 = data[3..6].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-5, "head 0 sum: {sum0}");
    assert!((sum1 - 1.0).abs() < 1e-5, "head 1 sum: {sum1}");
}

#[test]
fn softmax_metal_vs_cpu_agree() {
    let data = vec![0.5, -1.2, 3.0, 0.1, 2.2, -0.8, 1.5, 0.3, -2.0];
    let input = make_tensor(data.clone(), vec![3, 3]);

    // GPU
    let gpu_output = run_softmax(&input, 1);
    let gpu_data = read_f32(&gpu_output);

    // CPU reference
    let cpu_out = arrow_ml::activations::softmax::softmax(&input, 1).unwrap();
    let cpu_data = cpu_out.buffer().typed_data::<f32>().unwrap();

    assert_eq!(gpu_data.len(), cpu_data.len());
    for (i, (g, c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!((g - c).abs() < 1e-5, "mismatch at {i}: gpu={g}, cpu={c}");
    }
}
