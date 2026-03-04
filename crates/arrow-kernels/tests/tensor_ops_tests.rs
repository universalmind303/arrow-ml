use arrow::array::Float32Array;
use arrow::buffer::Buffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_kernels::array_ops::ArrayOps;
use arrow_kernels::tensor_ops::TensorOps;

fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
    Tensor::new_row_major(Buffer::from(data), Some(shape), None).unwrap()
}

// ---- TensorOps: linear algebra ----

#[test]
fn test_dot_matmul() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.dot(&b).unwrap();
    assert_eq!(c.shape().unwrap(), &[2, 2]);
    let data: &[f32] = c.data().typed_data();
    assert_eq!(data, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_transpose() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let t = a.t().unwrap();
    assert_eq!(t.shape().unwrap(), &[3, 2]);
    let data: &[f32] = t.data().typed_data();
    assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// ---- TensorOps: reshaping ----

#[test]
fn test_reshape() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = a.reshape(&[3, 2]).unwrap();
    assert_eq!(b.shape().unwrap(), &[3, 2]);
}

#[test]
fn test_flatten() {
    let a = make_f32(vec![1.0; 24], vec![2, 3, 4]);
    let b = a.flatten(1).unwrap();
    assert_eq!(b.shape().unwrap(), &[2, 12]);
}

#[test]
fn test_squeeze_unsqueeze() {
    let a = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
    let b = a.squeeze(None).unwrap();
    assert_eq!(b.shape().unwrap(), &[3]);
    let c = b.unsqueeze(&[0, 2]).unwrap();
    assert_eq!(c.shape().unwrap(), &[1, 3, 1]);
}

// ---- TensorOps: reductions ----

#[test]
fn test_sum() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = a.sum(&[1], false).unwrap();
    assert_eq!(s.data().typed_data::<f32>(), &[6.0, 15.0]);
}

#[test]
fn test_mean() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let m = a.mean(&[0], false).unwrap();
    assert_eq!(m.data().typed_data::<f32>(), &[2.5, 3.5, 4.5]);
}

#[test]
fn test_prod() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let p = a.prod(&[1], false).unwrap();
    let data: &[f32] = p.data().typed_data();
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 120.0).abs() < 1e-4);
}

// ---- TensorOps: elementwise math ----

#[test]
fn test_pow() {
    let a = make_f32(vec![2.0, 3.0, 4.0], vec![3]);
    let b = a.pow(2.0).unwrap();
    let data: &[f32] = b.data().typed_data();
    assert!((data[0] - 4.0).abs() < 1e-6);
    assert!((data[1] - 9.0).abs() < 1e-6);
    assert!((data[2] - 16.0).abs() < 1e-6);
}

#[test]
fn test_floor_ceil_round() {
    let a = make_f32(vec![1.3, 2.7, -0.5], vec![3]);
    assert_eq!(
        a.floor().unwrap().data().typed_data::<f32>(),
        &[1.0, 2.0, -1.0]
    );
    assert_eq!(
        a.ceil().unwrap().data().typed_data::<f32>(),
        &[2.0, 3.0, 0.0]
    );
}

#[test]
fn test_clip() {
    let a = make_f32(vec![-2.0, 0.5, 3.0], vec![3]);
    let b = a.clip(Some(0.0), Some(1.0)).unwrap();
    assert_eq!(b.data().typed_data::<f32>(), &[0.0, 0.5, 1.0]);
}

#[test]
fn test_reciprocal() {
    let a = make_f32(vec![2.0, 4.0, 5.0], vec![3]);
    let b = a.reciprocal().unwrap();
    let data: &[f32] = b.data().typed_data();
    assert!((data[0] - 0.5).abs() < 1e-6);
    assert!((data[1] - 0.25).abs() < 1e-6);
    assert!((data[2] - 0.2).abs() < 1e-6);
}

#[test]
fn test_cos_sin() {
    let a = make_f32(vec![0.0], vec![1]);
    let c = a.cos().unwrap();
    let s = a.sin().unwrap();
    assert!((c.data().typed_data::<f32>()[0] - 1.0).abs() < 1e-6);
    assert!((s.data().typed_data::<f32>()[0] - 0.0).abs() < 1e-6);
}

// ---- TensorOps: softmax ----

#[test]
fn test_softmax_tensor() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let s = a.softmax(-1).unwrap();
    assert_eq!(s.shape().unwrap(), &[2, 3]);
    let data: &[f32] = s.data().typed_data();
    let row0_sum: f32 = data[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-6);
}

// ---- TensorOps: chaining ----

#[test]
fn test_method_chaining() {
    // dot -> sum -> squeeze: verify chaining compiles and works
    let a = make_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = make_f32(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    let result = a.dot(&b).unwrap().sum(&[1], false).unwrap();
    assert_eq!(result.data().typed_data::<f32>(), &[7.0, 11.0]);
}

#[test]
fn test_dot_transpose_chain() {
    let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let at = a.t().unwrap();
    let result = at.dot(&a).unwrap();
    assert_eq!(result.shape().unwrap(), &[2, 2]);
    // A^T @ A = [[10, 14], [14, 20]]
    let data: &[f32] = result.data().typed_data();
    assert_eq!(data, &[10.0, 14.0, 14.0, 20.0]);
}

// ---- ArrayOps: activations on PrimitiveArray ----

#[test]
fn test_array_relu() {
    let arr = Float32Array::from(vec![1.0f32, -2.0, 3.0, -4.0]);
    let out = arr.relu();
    assert_eq!(out.values().as_ref(), &[1.0, 0.0, 3.0, 0.0]);
}

#[test]
fn test_array_sigmoid() {
    let arr = Float32Array::from(vec![0.0f32]);
    let out = arr.sigmoid();
    assert!((out.value(0) - 0.5).abs() < 1e-6);
}

#[test]
fn test_array_gelu() {
    let arr = Float32Array::from(vec![0.0f32, 1.0, -1.0]);
    let out = arr.gelu();
    assert!((out.value(0) - 0.0).abs() < 1e-5);
    assert!(out.value(1) > 0.0);
    assert!(out.value(2) < 0.0);
}

#[test]
fn test_array_softmax() {
    let arr = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
    let out = arr.softmax().unwrap();
    let sum: f32 = out.values().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_array_silu() {
    let arr = Float32Array::from(vec![0.0f32, 1.0]);
    let out = arr.silu();
    assert!((out.value(0) - 0.0).abs() < 1e-6);
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    assert!((out.value(1) - 0.7311).abs() < 1e-3);
}

#[test]
fn test_array_leaky_relu() {
    let arr = Float32Array::from(vec![1.0f32, -2.0]);
    let out = arr.leaky_relu(0.1);
    assert!((out.value(0) - 1.0).abs() < 1e-6);
    assert!((out.value(1) - (-0.2)).abs() < 1e-6);
}
