use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml::activations::layernorm::layer_norm;
use arrow_ml::common::BackendRegistry;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;

fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from(ArrowTensor::<Float32Type>::new_row_major(data.into(), Some(shape), None).unwrap())
}

fn main() {
    let backends = BackendRegistry::global().loaded_backends();
    println!("Loaded backends: {backends:?}");
    assert!(backends.contains(&"metal"));

    let input =
        make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]).to(Device::metal(0));

    let gamma = make_f32(vec![1.0; 4], vec![4]).to(Device::metal(0));
    let beta = make_f32(vec![0.0; 4], vec![4]).to(Device::metal(0));

    println!(
        "Input on {:?}, shape {:?}",
        input.device(),
        input.shape().unwrap()
    );

    let output = layer_norm(&input, &gamma, &beta, -1, 1e-5).unwrap();
    let data = output
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();

    println!("Row 0: {:?}", &data[0..4]);
    println!("Row 1: {:?}", &data[4..8]);

    let row0_mean: f32 = data[0..4].iter().sum::<f32>() / 4.0;
    let row1_mean: f32 = data[4..8].iter().sum::<f32>() / 4.0;
    println!("Row 0 mean: {row0_mean} (should be ~0)");
    println!("Row 1 mean: {row1_mean} (should be ~0)");

    assert!(row0_mean.abs() < 1e-4);
    assert!(row1_mean.abs() < 1e-4);

    println!("\nMetal layer norm passed!");
}
