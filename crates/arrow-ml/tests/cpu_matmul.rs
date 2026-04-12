use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use arrow_ml_linalg::matmul::matmul;

fn make_f32(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
    let buffer = ScalarBuffer::<f32>::from(data).into_inner();
    let tensor =
        ArrowTensor::<Float32Type>::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap();
    Tensor::from(tensor)
}

fn make_f64(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
    let buffer = ScalarBuffer::<f64>::from(data).into_inner();
    let tensor =
        ArrowTensor::<Float64Type>::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap();
    Tensor::from(tensor)
}

fn assert_f32_eq(tensor: &Tensor, expected: &[f32]) {
    let data = tensor.buffer().typed_data::<f32>().unwrap();
    assert_eq!(
        data.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        data.len(),
        expected.len()
    );
    for (i, (got, want)) in data.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "element {i}: got {got}, expected {want}"
        );
    }
}

fn assert_f64_eq(tensor: &Tensor, expected: &[f64]) {
    let data = tensor.buffer().typed_data::<f64>().unwrap();
    for (i, (got, want)) in data.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "element {i}: got {got}, expected {want}"
        );
    }
}

// --- basic correctness ---

#[test]
fn matmul_2x3_times_3x2() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = make_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape().unwrap(), &[2, 2]);
    assert_eq!(c.device(), Device::Cpu);
    assert_f32_eq(&c, &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_identity() {
    let eye = make_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let a = make_f32(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = matmul(&eye, &a).unwrap();
    assert_f32_eq(&c, &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_1x1() {
    let a = make_f32(vec![3.0], 1, 1);
    let b = make_f32(vec![7.0], 1, 1);
    let c = matmul(&a, &b).unwrap();
    assert_f32_eq(&c, &[21.0]);
}

#[test]
fn matmul_f64() {
    let a = make_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = make_f64(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape().unwrap(), &[2, 2]);
    assert_f64_eq(&c, &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_wide_output() {
    // [1,3] @ [3,4] = [1,4]
    let a = make_f32(vec![1.0, 2.0, 3.0], 1, 3);
    let b = make_f32(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        3,
        4,
    );
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.shape().unwrap(), &[1, 4]);
    assert_f32_eq(&c, &[1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn matmul_tall_output() {
    // [4,1] @ [1,1] = [4,1]
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
    let b = make_f32(vec![2.0], 1, 1);
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.shape().unwrap(), &[4, 1]);
    assert_f32_eq(&c, &[2.0, 4.0, 6.0, 8.0]);
}

// --- reference comparison ---

#[test]
fn matmul_matches_naive_reference() {
    let m = 4;
    let k = 5;
    let n = 3;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32 * 0.2).collect();

    let a = make_f32(a_data.clone(), m, k);
    let b = make_f32(b_data.clone(), k, n);
    let c = matmul(&a, &b).unwrap();

    // naive reference
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                expected[i * n + j] += a_data[i * k + p] * b_data[p * n + j];
            }
        }
    }

    assert_f32_eq(&c, &expected);
}

// --- error cases ---

#[test]
fn matmul_dimension_mismatch() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = make_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(matmul(&a, &b).is_err());
}

#[test]
fn matmul_dtype_mismatch() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = make_f64(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(matmul(&a, &b).is_err());
}

// --- output is on CPU ---

#[test]
fn matmul_output_on_cpu() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = make_f32(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.device(), Device::Cpu);
}

// --- result can bridge back to arrow ---

#[test]
fn matmul_result_converts_to_arrow_tensor() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = make_f32(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = matmul(&a, &b).unwrap();

    let arrow_c: ArrowTensor<'static, Float32Type> = c.try_into().unwrap();
    assert_eq!(arrow_c.shape().unwrap(), &[2, 2]);
    let data = arrow_c.data().typed_data::<f32>();
    assert!((data[0] - 19.0).abs() < 1e-5); // 1*5 + 2*7
    assert!((data[1] - 22.0).abs() < 1e-5); // 1*6 + 2*8
    assert!((data[2] - 43.0).abs() < 1e-5); // 3*5 + 4*7
    assert!((data[3] - 50.0).abs() < 1e-5); // 3*6 + 4*8
}
