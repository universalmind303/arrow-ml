# arrow-kernels

High-performance machine learning kernels built on [Apache Arrow](https://arrow.apache.org/), written in Rust.

`arrow-kernels` provides optimized tensor operations, linear algebra primitives, and neural network building blocks that operate directly on Arrow's in-memory format. It features a tiered execution strategy — naive loops for small inputs, SIMD for medium workloads, and pluggable GPU backends for large computations.

## Features

- **40+ operations** covering linear algebra, activations, normalization, convolution, pooling, and more
- **Arrow-native** — works directly with `arrow::tensor::Tensor` and `PrimitiveArray`, zero-copy where possible
- **Tiered dispatch** — automatically selects between naive, SIMD (`portable_simd`), and GPU paths based on input size
- **Pluggable GPU backends** — ships with an Apple Metal backend; extensible via a C ABI plugin system
- **Type-generic** — optimized fast paths for `f32`/`f64`, generic fallback for all Arrow numeric types
- **Null-propagating** — correctly handles nullable Arrow arrays throughout

## Crate Structure

```
arrow-kernels              # Unified public API with TensorOps / ArrayOps traits
├── arrow-kernels-linalg         # Linear algebra, reductions, reshaping, conv, pooling
├── arrow-kernels-activations    # Activation functions (ReLU, GELU, Sigmoid, etc.)
├── arrow-kernels-common         # Shared error types & backend plugin registry
└── arrow-kernels-backend-metal  # Metal GPU backend (macOS, cdylib)
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
arrow-kernels = { path = "crates/arrow-kernels" }
arrow = { version = "57.1.0", default-features = false }
```

### Tensor Operations

```rust
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_kernels::tensor_ops::TensorOps;

// Create 2x3 and 3x2 tensors
let a_buf = Buffer::from(ScalarBuffer::<f32>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into_inner());
let a = Tensor::new_row_major(a_buf, Some(vec![2, 3]), None).unwrap();

let b_buf = Buffer::from(ScalarBuffer::<f32>::from(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).into_inner());
let b = Tensor::new_row_major(b_buf, Some(vec![3, 2]), None).unwrap();

// Matrix multiply, transpose, then sum along last axis
let result = a.dot::<Float32Type>(&b)?
    .t::<Float32Type>()?
    .sum::<Float32Type>(&[-1], false)?;
```

### Activation Functions

```rust
use arrow::array::Float32Array;
use arrow_kernels::array_ops::ArrayOps;

let input = Float32Array::from(vec![Some(-1.0), Some(0.0), Some(1.0), None]);
let activated = input.relu();       // [0.0, 0.0, 1.0, null]
let gelu = input.gelu();            // GELU approximation
let sig = input.sigmoid();          // Sigmoid
```

### Direct Kernel Calls

```rust
use arrow_kernels_linalg::matmul::matmul;
use arrow_kernels_linalg::layernorm::layer_norm;
use arrow_kernels_linalg::conv::conv2d;

let c = matmul(&a, &b)?;            // C = A * B
```

## Operations

### Linear Algebra

`matmul`, `gemm`, `matvec`, `gemv`, `dot`, `axpy`, `scal`

### Activations

`relu`, `leaky_relu`, `gelu`, `gelu_exact`, `sigmoid`, `tanh`, `silu`, `softmax`

### Normalization

`layer_norm`, `rms_norm`, `batch_norm`, `group_norm`, `instance_norm`, `l1_norm`, `l2_norm`

### Reductions

`reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`, `reduce_prod`, `cumsum`, `argmax`, `argmin`, `topk`

### Tensor Manipulation

`reshape`, `flatten`, `squeeze`, `unsqueeze`, `expand`, `transpose`, `concat`, `gather`, `gather_elements`, `scatter_nd`, `pad`

### Convolution & Pooling

`conv2d`, `conv_transpose2d`, `avg_pool2d`, `max_pool2d`, `resize`

### Element-wise Math

`pow`, `erf`, `reciprocal`, `cos`, `sin`, `floor`, `ceil`, `round`, `clip`

### Other

`embedding`, `where_cond`

## GPU Backend

On macOS, the Metal backend accelerates `matmul` for large matrices automatically. The dispatch threshold is configurable but defaults to 256×256.

The backend plugin system discovers shared libraries at runtime matching the pattern `libarrow_kernels_backend_*`. To build the Metal backend:

```sh
cargo build --release -p arrow-kernels-backend-metal
```

Custom backends can be added by implementing the C ABI contract (see `arrow-kernels-common` for the expected symbols).

## Benchmarks

```sh
cargo bench -p arrow-kernels-linalg
```

Benchmarks cover matmul performance across sizes (16–4096) for both `f32` and `f64`, comparing naive vs SIMD vs GPU paths.

## Requirements

- **Rust nightly** (uses `portable_simd`)
- Arrow 57.1.0
- macOS for Metal GPU backend (optional)

## License

Apache-2.0
