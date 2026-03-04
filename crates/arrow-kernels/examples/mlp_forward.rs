/// Simple 2-layer MLP forward pass: Linear -> ReLU -> Linear -> Softmax
///
/// Demonstrates: TensorOps (dot, softmax), ArrayOps (relu, softmax)
///
/// Architecture:
///   input(4) -> Linear(4, 8) -> ReLU -> Linear(8, 3) -> Softmax -> argmax
///
/// Run: cargo run -p arrow-kernels --example mlp_forward
use arrow::array::Float32Array;
use arrow::buffer::Buffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_kernels::array_ops::ArrayOps;
use arrow_kernels::linalg::argmax::argmax;
use arrow_kernels::tensor_ops::TensorOps;

fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
    Tensor::new_row_major(Buffer::from(data), Some(shape), None).unwrap()
}

fn add_bias(tensor: &Tensor<'_, Float32Type>, bias: &[f32]) -> Tensor<'static, Float32Type> {
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
    println!("=== MLP Forward Pass ===\n");

    // --- Fake weights (in a real model these would be loaded) ---
    // Layer 1: 4 -> 8
    let w1 = make_tensor(
        vec![
            0.1, 0.2, -0.1, 0.3, 0.05, -0.2, 0.15, 0.1, -0.3, 0.1, 0.2, -0.1, 0.4, 0.1, -0.05, 0.2,
            0.2, -0.15, 0.3, 0.1, -0.1, 0.25, 0.1, -0.2, -0.1, 0.3, -0.2, 0.15, 0.2, -0.1, 0.3,
            0.05,
        ],
        vec![4, 8],
    );
    let b1 = vec![0.1, -0.1, 0.05, 0.0, 0.1, -0.05, 0.0, 0.1];

    // Layer 2: 8 -> 3
    let w2 = make_tensor(
        vec![
            0.2, -0.1, 0.3, 0.1, 0.15, -0.2, 0.05, -0.1, -0.1, 0.3, -0.15, 0.2, 0.1, 0.2, -0.3,
            0.1, 0.15, -0.2, 0.1, 0.25, -0.1, 0.05, 0.3, -0.15,
        ],
        vec![8, 3],
    );
    let b2 = vec![0.0, 0.1, -0.1];

    // --- Batch of 2 input samples, each with 4 features ---
    let input = make_tensor(vec![1.0, 0.5, -0.3, 0.8, -0.2, 1.2, 0.4, -0.6], vec![2, 4]);
    println!("Input (2 samples, 4 features):");
    print_tensor(&input);

    // --- Layer 1: Linear + ReLU ---
    let hidden = input.dot(&w1).unwrap();
    let hidden = add_bias(&hidden, &b1);
    println!("\nAfter Linear layer 1 (2x8):");
    print_tensor(&hidden);

    // Apply ReLU via ArrayOps on the flat data
    let hidden_data: &[f32] = hidden.data().typed_data();
    let hidden_arr = Float32Array::from_iter_values(hidden_data.iter().copied());
    let hidden_relu = hidden_arr.relu();
    let hidden = make_tensor(
        hidden_relu.values().to_vec(),
        hidden.shape().unwrap().to_vec(),
    );
    println!("\nAfter ReLU:");
    print_tensor(&hidden);

    // --- Layer 2: Linear ---
    let logits = hidden.dot(&w2).unwrap();
    let logits = add_bias(&logits, &b2);
    println!("\nLogits (2x3):");
    print_tensor(&logits);

    // --- Softmax per sample via TensorOps ---
    let probs = logits.softmax(-1).unwrap();
    let probs_data: &[f32] = probs.data().typed_data();
    let n_classes = 3;
    println!("\nProbabilities & predictions:");
    for i in 0..2 {
        let row = &probs_data[i * n_classes..(i + 1) * n_classes];
        let row_arr = Float32Array::from_iter_values(row.iter().copied());
        let pred = argmax(&row_arr).unwrap();
        println!(
            "  Sample {}: probs=[{:.4}, {:.4}, {:.4}] -> class {}",
            i, row[0], row[1], row[2], pred
        );
    }

    println!("\nDone!");
}

fn print_tensor(tensor: &Tensor<'_, Float32Type>) {
    let shape = tensor.shape().unwrap();
    let data: &[f32] = tensor.data().typed_data();
    if shape.len() == 2 {
        let cols = shape[1];
        for row in data.chunks(cols) {
            let vals: Vec<String> = row.iter().map(|v| format!("{:>8.4}", v)).collect();
            println!("  [{}]", vals.join(", "));
        }
    } else {
        println!("  {:?}", data);
    }
}
