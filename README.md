# arrow-ml

Device-aware, [Arrow](https://arrow.apache.org/)-compatible data types and ML kernels, written in Rust.

`arrow-ml` provides two things: device-aware types that wrap Arrow's in-memory arrays and tensors so they can live on CPU or GPU, and a set of ML kernels that operate on them. You decide where data lives — `.to(device)` moves it, and kernels execute wherever their inputs are.

## Features

- **Arrow-native** — zero-copy conversion from `arrow::tensor::Tensor` and `PrimitiveArray` into device-aware types
- **Explicit device placement** — `.to(Device::metal(0))` moves to GPU, `.to(Device::cpu())` brings it back
- **Pluggable GPU backends** — ships with an Apple Metal backend; extensible via a C ABI plugin system
- **Type-generic** — optimized fast paths for `f32`/`f64`
- **Null-propagating** — `DeviceArray` preserves nullable Arrow arrays throughout

## Crate Structure

```
arrow-ml                    # Re-exports: activations, linalg, common
├── arrow-ml-core           # Device-aware types: Tensor, DeviceArray, DeviceBuffer, Device
├── arrow-ml-linalg         # Linear algebra (matmul)
├── arrow-ml-activations    # Activation functions (ReLU, GELU, Sigmoid, etc.)
└── arrow-ml-common         # Shared error types, backend plugin registry, FFI types

arrow-ml-backend-metal      # Metal GPU backend (macOS, cdylib) — standalone crate
```

`arrow-ml-backend-metal` sits outside the main workspace as an independent
crate with its own release pipeline so that the cross-platform crates stay
buildable on every target.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
arrow-ml = "0.1"
arrow-ml-core = "0.1"
arrow = { version = ">=56, <59", default-features = false }
```

### CPU Matmul

```rust
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_core::tensor::Tensor;
use arrow_ml_linalg::matmul::matmul;

// Zero-copy from Arrow tensors
let a = Tensor::from(
    ArrowTensor::<Float32Type>::new_row_major(
        ScalarBuffer::<f32>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into_inner(),
        Some(vec![2, 3]), None,
    ).unwrap(),
);
let b = Tensor::from(
    ArrowTensor::<Float32Type>::new_row_major(
        ScalarBuffer::<f32>::from(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).into_inner(),
        Some(vec![3, 2]), None,
    ).unwrap(),
);

let c = matmul(&a, &b)?;  // [2, 2] on CPU
```

### Moving Data to GPU

```rust
use arrow_ml_core::device::Device;

let a_gpu = a.to(Device::metal(0));
let b_gpu = b.to(Device::metal(0));
let c_gpu = matmul(&a_gpu, &b_gpu)?;  // runs on GPU, result stays on GPU
let c_cpu = c_gpu.to(Device::cpu());   // bring back to host
```

### Bridging Back to Arrow

```rust

// This will error if you have not first moved the data back to cpu.
assert!(a_gpu.try_into::<ArrowTensor<_, _>>().is_err());

let arrow_tensor: ArrowTensor<'static, Float32Type> = c_cpu.try_into()?;
```

## GPU Backend

Kernels run on whatever device their inputs are on. Move data to GPU with
`.to(Device::metal(0))`, and the GPU kernel runs. Both operands must be on
the same device or the kernel returns an error.

The backend plugin system discovers backends at runtime via two parallel
mechanisms:

1. **JSON manifests** (Vulkan-ICD style) — `*.json` files in
   `~/.arrow-ml/backends/`, `/etc/arrow-ml/backends/`, or
   `$ARROW_ML_BACKEND_MANIFEST_DIR`. A reference manifest ships at
   `crates/arrow-ml-backend-metal/manifests/metal.json`.
2. **Glob fallback** — shared libraries matching `libarrow_ml_backend_*`
   sitting next to the running executable, in `$ARROW_ML_BACKEND_DIR`, or
   in the workspace `target/{debug,release}` during development.

The Metal backend lives in its own standalone crate (not part of the cargo
workspace) so that the rest of arrow-ml stays cross-platform. To build it:

```sh
cd crates/arrow-ml-backend-metal
cargo build --release
```

Or, as a downstream consumer, just add `arrow-ml-backend-metal = "0.1"` to
your `Cargo.toml` alongside `arrow-ml` — Cargo will produce the cdylib in
your `target/` directory and arrow-ml's runtime registry will discover it.

Custom backends can be added by implementing the C ABI contract — every
backend must export `am_backend_abi_version`, `am_backend_name`, and
`am_backend_priority`. See `arrow_ml_common::backend` for the full
function-pointer types and the current `ARROW_ML_BACKEND_ABI_VERSION`.

## Benchmarks

```sh
cargo bench -p arrow-ml-linalg
```

Benchmarks cover matmul performance across sizes for `f32` and `f64`.

## Requirements

- **Rust nightly** (uses `portable_simd`)
- Arrow 56–58
- macOS for Metal GPU backend (optional)

## License

Apache-2.0
