/// Mini transformer block: Embedding -> Self-Attention -> LayerNorm -> FFN
///
/// Demonstrates: TensorOps (dot, t, softmax, erf), ArrayOps (softmax)
///
/// Architecture (single sequence, 1 head for clarity):
///   tokens -> Embedding(vocab=16, dim=8) -> SelfAttention(Q,K,V) -> LayerNorm
///   -> FFN(8->16->8) with GELU -> LayerNorm -> argmax predictions
///
/// Run: cargo run -p arrow-kernels --example transformer_block
use arrow::array::{Float32Array, PrimitiveArray, UInt32Array};
use arrow::buffer::Buffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_kernels::array_ops::ArrayOps;
use arrow_kernels::linalg::argmax::{argmax, argmax_tensor};
use arrow_kernels::linalg::embedding::embedding;
use arrow_kernels::linalg::layernorm::layer_norm;
use arrow_kernels::tensor_ops::TensorOps;

fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
    Tensor::new_row_major(Buffer::from(data), Some(shape), None).unwrap()
}

fn make_array(data: Vec<f32>) -> PrimitiveArray<Float32Type> {
    Float32Array::from(data)
}

/// Element-wise: tensor + tensor (same shape)
fn add_tensors(
    a: &Tensor<'_, Float32Type>,
    b: &Tensor<'_, Float32Type>,
) -> Tensor<'static, Float32Type> {
    let shape = a.shape().unwrap().to_vec();
    let a_data: &[f32] = a.data().typed_data();
    let b_data: &[f32] = b.data().typed_data();
    let out: Vec<f32> = a_data.iter().zip(b_data).map(|(x, y)| x + y).collect();
    make_tensor(out, shape)
}

/// Element-wise: tensor * scalar
fn scale_tensor(t: &Tensor<'_, Float32Type>, s: f32) -> Tensor<'static, Float32Type> {
    let shape = t.shape().unwrap().to_vec();
    let data: Vec<f32> = t
        .data()
        .typed_data::<f32>()
        .iter()
        .map(|&x| x * s)
        .collect();
    make_tensor(data, shape)
}

/// Element-wise: tensor * tensor (same shape)
fn mul_tensors(
    a: &Tensor<'_, Float32Type>,
    b: &Tensor<'_, Float32Type>,
) -> Tensor<'static, Float32Type> {
    let a_data: &[f32] = a.data().typed_data();
    let b_data: &[f32] = b.data().typed_data();
    let out: Vec<f32> = a_data.iter().zip(b_data).map(|(x, y)| x * y).collect();
    make_tensor(out, a.shape().unwrap().to_vec())
}

/// Add bias to each row: (seq, dim) + (dim,)
fn add_bias(tensor: &Tensor<'_, Float32Type>, bias: &[f32]) -> Tensor<'static, Float32Type> {
    let shape = tensor.shape().unwrap().to_vec();
    let cols = shape[1];
    let data: &[f32] = tensor.data().typed_data();
    let out: Vec<f32> = data
        .chunks(cols)
        .flat_map(|row| row.iter().zip(bias).map(|(x, b)| x + b))
        .collect();
    make_tensor(out, shape)
}

/// GELU activation using TensorOps::erf: 0.5 * x * (1 + erf(x / sqrt(2)))
fn gelu_tensor(x: &Tensor<'_, Float32Type>) -> Tensor<'static, Float32Type> {
    let shape = x.shape().unwrap().to_vec();
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    let scaled = scale_tensor(x, inv_sqrt2);
    let erf_out = scaled.erf().unwrap();

    // 1 + erf(x/sqrt(2))
    let erf_data: &[f32] = erf_out.data().typed_data();
    let one_plus_erf: Vec<f32> = erf_data.iter().map(|&v| 1.0 + v).collect();
    let one_plus_erf = make_tensor(one_plus_erf, shape);

    // 0.5 * x * (1 + erf(...))
    let half_x = scale_tensor(x, 0.5);
    mul_tensors(&half_x, &one_plus_erf)
}

fn main() {
    println!("=== Mini Transformer Block ===\n");

    let vocab_size = 16;
    let embed_dim = 8;
    let ffn_hidden = 16;
    let seq_len = 4;

    // --- Token IDs ---
    let tokens = UInt32Array::from(vec![3u32, 7, 1, 12]);
    println!("Input tokens: [3, 7, 1, 12]  (vocab={})\n", vocab_size);

    // --- Embedding table (vocab_size x embed_dim) ---
    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| ((i as f32 * 0.17) - 1.0).sin() * 0.5)
        .collect();
    let embed_table = make_tensor(embed_data, vec![vocab_size, embed_dim]);

    let x = embedding(&embed_table, &tokens).unwrap();
    println!("After Embedding: shape {:?}", x.shape().unwrap());
    print_tensor_summary(&x, "  ");

    // =============================================================
    // Self-Attention: Q = xW_q, K = xW_k, V = xW_v
    //   attn = softmax(Q @ K^T / sqrt(d_k)) @ V
    // =============================================================
    let wq_data: Vec<f32> = (0..embed_dim * embed_dim)
        .map(|i| ((i as f32 * 0.13) - 0.5).cos() * 0.3)
        .collect();
    let wk_data: Vec<f32> = (0..embed_dim * embed_dim)
        .map(|i| ((i as f32 * 0.19) + 0.3).sin() * 0.3)
        .collect();
    let wv_data: Vec<f32> = (0..embed_dim * embed_dim)
        .map(|i| ((i as f32 * 0.23) - 0.7).cos() * 0.3)
        .collect();

    let w_q = make_tensor(wq_data, vec![embed_dim, embed_dim]);
    let w_k = make_tensor(wk_data, vec![embed_dim, embed_dim]);
    let w_v = make_tensor(wv_data, vec![embed_dim, embed_dim]);

    let q = x.dot(&w_q).unwrap();
    let k = x.dot(&w_k).unwrap();
    let v = x.dot(&w_v).unwrap();
    println!("\nQ, K, V computed: each {:?}", q.shape().unwrap());

    // Attention scores: Q @ K^T -> (seq, seq)
    let k_t = k.t().unwrap();
    let scores = q.dot(&k_t).unwrap();
    let scale = 1.0 / (embed_dim as f32).sqrt();
    let scores = scale_tensor(&scores, scale);
    println!(
        "Attention scores (scaled): shape {:?}",
        scores.shape().unwrap()
    );

    // Softmax over last axis
    let attn_weights = scores.softmax(-1).unwrap();
    println!("Attention weights (after softmax):");
    print_tensor_summary(&attn_weights, "  ");

    // Weighted sum: attn @ V -> (seq, dim)
    let attn_out = attn_weights.dot(&v).unwrap();

    // --- Residual + LayerNorm ---
    let residual1 = add_tensors(&x, &attn_out);
    let ln_gamma = make_array(vec![1.0; embed_dim]);
    let ln_beta = make_array(vec![0.0; embed_dim]);
    let normed1 = layer_norm(&residual1, &ln_gamma, &ln_beta, 1e-5).unwrap();
    println!(
        "\nAfter Attention + Residual + LayerNorm: shape {:?}",
        normed1.shape().unwrap()
    );
    print_tensor_summary(&normed1, "  ");

    // =============================================================
    // FFN: Linear(dim -> ffn_hidden) -> GELU -> Linear(ffn_hidden -> dim)
    // =============================================================
    let w_ff1_data: Vec<f32> = (0..embed_dim * ffn_hidden)
        .map(|i| ((i as f32 * 0.11) - 0.3).sin() * 0.25)
        .collect();
    let b_ff1: Vec<f32> = (0..ffn_hidden).map(|i| (i as f32 * 0.01) - 0.05).collect();
    let w_ff2_data: Vec<f32> = (0..ffn_hidden * embed_dim)
        .map(|i| ((i as f32 * 0.07) + 0.2).cos() * 0.25)
        .collect();
    let b_ff2: Vec<f32> = (0..embed_dim).map(|i| (i as f32 * 0.005) - 0.02).collect();

    let w_ff1 = make_tensor(w_ff1_data, vec![embed_dim, ffn_hidden]);
    let w_ff2 = make_tensor(w_ff2_data, vec![ffn_hidden, embed_dim]);

    let hidden = normed1.dot(&w_ff1).unwrap();
    let hidden = add_bias(&hidden, &b_ff1);
    let hidden = gelu_tensor(&hidden);
    println!(
        "FFN hidden (after GELU): shape {:?}",
        hidden.shape().unwrap()
    );

    let ffn_out = hidden.dot(&w_ff2).unwrap();
    let ffn_out = add_bias(&ffn_out, &b_ff2);

    // --- Residual + LayerNorm ---
    let residual2 = add_tensors(&normed1, &ffn_out);
    let ln2_gamma = make_array(vec![1.0; embed_dim]);
    let ln2_beta = make_array(vec![0.0; embed_dim]);
    let output = layer_norm(&residual2, &ln2_gamma, &ln2_beta, 1e-5).unwrap();
    println!(
        "After FFN + Residual + LayerNorm: shape {:?}",
        output.shape().unwrap()
    );
    print_tensor_summary(&output, "  ");

    // --- Predict next token: project to vocab and argmax ---
    let head_data: Vec<f32> = (0..embed_dim * vocab_size)
        .map(|i| ((i as f32 * 0.09) - 0.4).sin() * 0.3)
        .collect();
    let head = make_tensor(head_data, vec![embed_dim, vocab_size]);
    let logits = output.dot(&head).unwrap();

    // Argmax over vocab dimension for each position
    let predictions = argmax_tensor(&logits, -1, false).unwrap();
    let pred_data: &[i64] = predictions.data().typed_data();
    println!("\nPer-position predicted token IDs: {:?}", pred_data);

    // Show softmax probs for last position (next-token prediction)
    let logits_data: &[f32] = logits.data().typed_data();
    let last_logits = &logits_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    let last_arr = Float32Array::from_iter_values(last_logits.iter().copied());
    let last_probs = last_arr.softmax().unwrap();
    let top_token = argmax(&last_probs).unwrap();
    println!(
        "Next-token prediction (from position {}): token {} (prob={:.4})",
        seq_len - 1,
        top_token,
        last_probs.value(top_token)
    );

    println!("\nDone!");
}

fn print_tensor_summary(tensor: &Tensor<'_, Float32Type>, prefix: &str) {
    let shape = tensor.shape().unwrap();
    let data: &[f32] = tensor.data().typed_data();
    if shape.len() == 2 {
        let cols = shape[1];
        for (i, row) in data.chunks(cols).enumerate() {
            if i >= 3 && i < shape[0] - 1 {
                if i == 3 {
                    println!("{}  ...", prefix);
                }
                continue;
            }
            let vals: Vec<String> = row.iter().take(6).map(|v| format!("{:>7.4}", v)).collect();
            println!("{}[{}]", prefix, vals.join(", "));
        }
    }
}
