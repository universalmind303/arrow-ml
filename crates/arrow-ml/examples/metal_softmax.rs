use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml::activations::softmax::softmax;
use arrow_ml::common::BackendRegistry;
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

    // 1D softmax
    let input_1d = Tensor::from(
        ArrowTensor::<Float32Type>::new_row_major(
            vec![1.0f32, 2.0, 3.0, 4.0].into(),
            Some(vec![4]),
            None,
        )
        .unwrap(),
    );

    let out = softmax(&input_1d.to(Device::metal(0)), 0).unwrap();
    let data = out
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    let sum: f32 = data.iter().sum();
    println!("1D softmax([1,2,3,4]): {data:?}");
    println!("  sum = {sum}");
    assert!((sum - 1.0).abs() < 1e-5);

    // 2D softmax over rows (last axis) — attention-style
    let input_2d = Tensor::from(
        ArrowTensor::<Float32Type>::new_row_major(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].into(),
            Some(vec![2, 3]),
            None,
        )
        .unwrap(),
    );

    let out = softmax(&input_2d.to(Device::metal(0)), -1).unwrap();
    let data = out
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    let row0_sum: f32 = data[0..3].iter().sum();
    let row1_sum: f32 = data[3..6].iter().sum();
    println!("2D softmax (axis=-1):");
    println!("  row 0: {:?}  sum={row0_sum}", &data[0..3]);
    println!("  row 1: {:?}  sum={row1_sum}", &data[3..6]);
    assert!((row0_sum - 1.0).abs() < 1e-5);
    assert!((row1_sum - 1.0).abs() < 1e-5);

    // 3D softmax — (batch=1, heads=2, seq=4), softmax over seq dim
    let input_3d = Tensor::from(
        ArrowTensor::<Float32Type>::new_row_major(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into(),
            Some(vec![1, 2, 4]),
            None,
        )
        .unwrap(),
    );

    let out = softmax(&input_3d.to(Device::metal(0)), -1).unwrap();
    let data = out
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    let head0_sum: f32 = data[0..4].iter().sum();
    let head1_sum: f32 = data[4..8].iter().sum();
    println!("3D attention softmax (1,2,4) axis=-1:");
    println!("  head 0: {:?}  sum={head0_sum}", &data[0..4]);
    println!("  head 1: {:?}  sum={head1_sum}", &data[4..8]);
    assert!((head0_sum - 1.0).abs() < 1e-5);
    assert!((head1_sum - 1.0).abs() < 1e-5);

    println!("\nMetal softmax passed!");
}
