#![cfg(target_os = "macos")]

use arrow::buffer::ScalarBuffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml::activations::gelu::gelu;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;

fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let arrow_t = ArrowTensor::<Float32Type>::new_row_major(
        ScalarBuffer::<f32>::from(data).into_inner(),
        Some(shape),
        None,
    )
    .unwrap();
    Tensor::from(arrow_t)
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    t.to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
fn gelu_zero() {
    let input = make_f32(vec![0.0], vec![1]).to(Device::metal(0));
    let output = gelu(&input).unwrap();
    let data = read_f32(&output);
    assert!(data[0].abs() < 1e-6);
}

#[test]
fn gelu_positive_values() {
    let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]).to(Device::metal(0));
    let output = gelu(&input).unwrap();
    let data = read_f32(&output);
    // All positive inputs should produce positive outputs
    for &v in &data {
        assert!(v > 0.0);
    }
    // Monotonically increasing for positive inputs
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);
}

#[test]
fn gelu_large_positive_approx_identity() {
    let input = make_f32(vec![4.0, 5.0], vec![2]).to(Device::metal(0));
    let output = gelu(&input).unwrap();
    let data = read_f32(&output);
    // For large-ish x, gelu(x) ≈ x
    assert!((data[0] - 4.0).abs() < 1e-3, "gelu(4) = {}", data[0]);
    assert!((data[1] - 5.0).abs() < 1e-3, "gelu(5) = {}", data[1]);
}

#[test]
fn gelu_metal_vs_cpu_agree() {
    let vals = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let input = make_f32(vals.clone(), vec![9]);

    let cpu_out = gelu(&input).unwrap();
    let cpu_data = cpu_out.buffer().typed_data::<f32>().unwrap().to_vec();

    let gpu_out = gelu(&input.to(Device::metal(0))).unwrap();
    let gpu_data = read_f32(&gpu_out);

    assert_eq!(cpu_data.len(), gpu_data.len());
    for (i, (c, g)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
        assert!((c - g).abs() < 1e-5, "mismatch at {i}: cpu={c}, gpu={g}");
    }
}

#[test]
fn gelu_preserves_shape_on_metal() {
    let input = make_f32(vec![1.0; 12], vec![3, 4]).to(Device::metal(0));
    let output = gelu(&input).unwrap();
    assert_eq!(output.shape().unwrap(), &[3, 4]);
    assert_eq!(output.device(), Device::metal(0));
}
