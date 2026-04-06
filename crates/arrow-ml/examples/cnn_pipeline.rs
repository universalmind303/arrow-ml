/// Mini CNN inference pipeline: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear
///
/// Demonstrates: TensorOps (dot, flatten, softmax), ArrayOps (relu, softmax)
///
/// Architecture (single 8x8 grayscale image):
///   input(1,1,8,8) -> Conv2D(1->4, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
///   -> Flatten -> Linear(4*3*3, 3) -> Softmax
///
/// Run: cargo run -p arrow-ml --example cnn_pipeline
use arrow::array::{Float32Array, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_ml::array_ops::ArrayOps;
use arrow_ml::linalg::argmax::argmax;
use arrow_ml::linalg::batchnorm::batch_norm;
use arrow_ml::linalg::conv::conv2d;
use arrow_ml::linalg::pooling::max_pool2d;
use arrow_ml::tensor_ops::TensorOps;

fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
    Tensor::new_row_major(Buffer::from(data), Some(shape), None).unwrap()
}

fn make_array(data: Vec<f32>) -> PrimitiveArray<Float32Type> {
    Float32Array::from(data)
}

fn relu_tensor(t: &Tensor<'_, Float32Type>) -> Tensor<'static, Float32Type> {
    let shape = t.shape().unwrap().to_vec();
    let data: Vec<f32> = t
        .data()
        .typed_data::<f32>()
        .iter()
        .map(|&x| x.max(0.0))
        .collect();
    make_tensor(data, shape)
}

fn add_bias_1d(tensor: &Tensor<'_, Float32Type>, bias: &[f32]) -> Tensor<'static, Float32Type> {
    let shape = tensor.shape().unwrap();
    let cols = shape[1];
    let data: &[f32] = tensor.data().typed_data();
    let out: Vec<f32> = data
        .chunks(cols)
        .flat_map(|row| row.iter().zip(bias).map(|(x, b)| x + b))
        .collect();
    make_tensor(out, shape.to_vec())
}

fn main() {
    println!("=== CNN Inference Pipeline ===\n");

    // --- Synthetic 8x8 grayscale "image" (1 batch, 1 channel) ---
    let image_data: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
    let input = make_tensor(image_data, vec![1, 1, 8, 8]);
    println!("Input: 1x1x8x8 grayscale image (values 0..1)");

    // --- Conv2D: 1 input channel -> 4 output channels, 3x3 kernel ---
    let conv_weight_data: Vec<f32> = (0..4 * 1 * 3 * 3)
        .map(|i| ((i as f32 * 0.1) - 0.5).sin() * 0.3)
        .collect();
    let conv_weight = make_tensor(conv_weight_data, vec![4, 1, 3, 3]);
    let conv_bias = make_array(vec![0.1, -0.05, 0.0, 0.05]);

    let conv_out = conv2d(
        &input,
        &conv_weight,
        Some(&conv_bias),
        [1, 1], // padding
        [1, 1], // stride
        [1, 1], // dilation
        1,      // groups
    )
    .unwrap();
    println!(
        "After Conv2D(1->4, 3x3, pad=1): shape {:?}",
        conv_out.shape().unwrap()
    );

    // --- BatchNorm ---
    let bn_scale = make_array(vec![1.0; 4]);
    let bn_bias = make_array(vec![0.0; 4]);
    let bn_mean = make_array(vec![0.0; 4]);
    let bn_var = make_array(vec![1.0; 4]);

    let bn_out = batch_norm(&conv_out, &bn_scale, &bn_bias, &bn_mean, &bn_var, 1e-5).unwrap();
    println!("After BatchNorm: shape {:?}", bn_out.shape().unwrap());

    // --- ReLU ---
    let relu_out = relu_tensor(&bn_out);
    println!("After ReLU: shape {:?}", relu_out.shape().unwrap());

    // --- MaxPool2D: 2x2, stride 2 ---
    let pool_out = max_pool2d(&relu_out, [2, 2], [2, 2], [0, 0]).unwrap();
    println!(
        "After MaxPool2D(2x2, stride=2): shape {:?}",
        pool_out.shape().unwrap()
    );

    // --- Flatten to 2D: (batch, features) ---
    let flat = pool_out.flatten(1).unwrap();
    println!("After Flatten: shape {:?}", flat.shape().unwrap());
    let n_features = flat.shape().unwrap()[1];

    // --- Linear: features -> 3 classes ---
    let fc_weight_data: Vec<f32> = (0..n_features * 3)
        .map(|i| ((i as f32 * 0.3) - 1.0).cos() * 0.2)
        .collect();
    let fc_weight = make_tensor(fc_weight_data, vec![n_features, 3]);
    let fc_bias = vec![0.0, 0.1, -0.1];

    let logits = flat.dot(&fc_weight).unwrap();
    let logits = add_bias_1d(&logits, &fc_bias);
    println!(
        "After Linear({}->3): shape {:?}",
        n_features,
        logits.shape().unwrap()
    );

    // --- Softmax + prediction ---
    let logits_data: &[f32] = logits.data().typed_data();
    let logits_arr = Float32Array::from_iter_values(logits_data.iter().copied());
    let probs = logits_arr.softmax().unwrap();
    let pred = argmax(&probs).unwrap();

    println!(
        "\nLogits:        [{:.4}, {:.4}, {:.4}]",
        logits_data[0], logits_data[1], logits_data[2]
    );
    println!(
        "Probabilities: [{:.4}, {:.4}, {:.4}]",
        probs.value(0),
        probs.value(1),
        probs.value(2)
    );
    println!("Predicted class: {}", pred);
    println!("\nDone!");
}
