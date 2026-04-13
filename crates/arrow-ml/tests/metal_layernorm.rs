#![cfg(target_os = "macos")]

use arrow::buffer::ScalarBuffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml::activations::layernorm::layer_norm;
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

fn ones(n: usize) -> Tensor {
    make_f32(vec![1.0; n], vec![n]).to(Device::metal(0))
}

fn zeros(n: usize) -> Tensor {
    make_f32(vec![0.0; n], vec![n]).to(Device::metal(0))
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    t.to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
fn layernorm_identity_params() {
    let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).to(Device::metal(0));
    let output = layer_norm(&input, &ones(5), &zeros(5), -1, 1e-5).unwrap();
    let data = read_f32(&output);

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
}

#[test]
fn layernorm_scale_shift() {
    let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).to(Device::metal(0));
    let gamma = make_f32(vec![2.0; 4], vec![4]).to(Device::metal(0));
    let beta = make_f32(vec![1.0; 4], vec![4]).to(Device::metal(0));
    let output = layer_norm(&input, &gamma, &beta, -1, 1e-5).unwrap();
    let data = read_f32(&output);

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "mean should be ~1.0 after shift, got {mean}"
    );
}

#[test]
fn layernorm_2d_rows() {
    let input = make_f32(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]).to(Device::metal(0));
    let output = layer_norm(&input, &ones(3), &zeros(3), -1, 1e-5).unwrap();
    let data = read_f32(&output);

    let row0_mean: f32 = data[0..3].iter().sum::<f32>() / 3.0;
    let row1_mean: f32 = data[3..6].iter().sum::<f32>() / 3.0;
    assert!(row0_mean.abs() < 1e-4, "row 0 mean: {row0_mean}");
    assert!(row1_mean.abs() < 1e-4, "row 1 mean: {row1_mean}");
}

#[test]
fn layernorm_constant_input() {
    let input = make_f32(vec![5.0; 4], vec![1, 4]).to(Device::metal(0));
    let output = layer_norm(&input, &ones(4), &zeros(4), -1, 1e-5).unwrap();
    let data = read_f32(&output);

    for v in &data {
        assert!(
            v.abs() < 1e-3,
            "constant input should normalize to ~0, got {v}"
        );
    }
}

#[test]
fn layernorm_metal_vs_cpu_agree() {
    let data = vec![0.5, -1.2, 3.0, 0.1, 2.2, -0.8, 1.5, 0.3, -2.0];
    let gamma_data = vec![1.5, 0.8, 2.0];
    let beta_data = vec![0.1, -0.5, 0.3];

    let input = make_f32(data.clone(), vec![3, 3]);
    let gamma = make_f32(gamma_data.clone(), vec![3]);
    let beta = make_f32(beta_data.clone(), vec![3]);

    // CPU
    let cpu_out = layer_norm(&input, &gamma, &beta, -1, 1e-5).unwrap();
    let cpu_data = cpu_out.buffer().typed_data::<f32>().unwrap().to_vec();

    // GPU
    let gpu_out = layer_norm(
        &input.to(Device::metal(0)),
        &gamma.to(Device::metal(0)),
        &beta.to(Device::metal(0)),
        -1,
        1e-5,
    )
    .unwrap();
    let gpu_data = read_f32(&gpu_out);

    assert_eq!(cpu_data.len(), gpu_data.len());
    for (i, (c, g)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
        assert!((c - g).abs() < 1e-4, "mismatch at {i}: cpu={c}, gpu={g}");
    }
}

#[test]
fn layernorm_3d_transformer_pattern() {
    // (batch=1, seq=2, hidden=4), normalize over hidden dim
    let input =
        make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 4]).to(Device::metal(0));
    let output = layer_norm(&input, &ones(4), &zeros(4), -1, 1e-5).unwrap();
    let data = read_f32(&output);

    let row0_mean: f32 = data[0..4].iter().sum::<f32>() / 4.0;
    let row1_mean: f32 = data[4..8].iter().sum::<f32>() / 4.0;
    assert!(row0_mean.abs() < 1e-4, "row 0 mean: {row0_mean}");
    assert!(row1_mean.abs() < 1e-4, "row 1 mean: {row1_mean}");
}
