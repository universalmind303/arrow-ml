use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml::common::BackendRegistry;
use arrow_ml::linalg::matmul::matmul;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;

fn main() {
    let registry = BackendRegistry::global();
    let backends = registry.loaded_backends();
    println!("Loaded backends: {backends:?}");

    assert!(
        backends.contains(&"metal"),
        "Metal backend not found. Make sure libarrow_ml_backend_metal.dylib is built."
    );

    // A: 2x3
    let a = Tensor::from(
        ArrowTensor::<Float32Type>::new_row_major(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].into(),
            Some(vec![2, 3]),
            None,
        )
        .unwrap(),
    );

    // B: 3x2
    let b = Tensor::from(
        ArrowTensor::<Float32Type>::new_row_major(
            vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0].into(),
            Some(vec![3, 2]),
            None,
        )
        .unwrap(),
    );

    // Move to Metal
    let a_metal = a.to(Device::metal(0));
    let b_metal = b.to(Device::metal(0));
    println!(
        "A on {:?}, shape {:?}",
        a_metal.device(),
        a_metal.shape().unwrap()
    );
    println!(
        "B on {:?}, shape {:?}",
        b_metal.device(),
        b_metal.shape().unwrap()
    );

    // Compute C = A @ B on Metal
    let c_metal = matmul(&a_metal, &b_metal).expect("matmul failed");
    println!(
        "C on {:?}, shape {:?}",
        c_metal.device(),
        c_metal.shape().unwrap()
    );

    // Copy result back to host to read values
    let c_host = c_metal.to(Device::cpu());
    let result = c_host.buffer().typed_data::<f32>().unwrap();

    // Expected: [[58, 64], [139, 154]]
    println!("Result: {result:?}");
    assert_eq!(result, &[58.0, 64.0, 139.0, 154.0]);
    println!("Metal matmul passed!");
}
