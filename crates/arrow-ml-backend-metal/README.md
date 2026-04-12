# arrow-ml-backend-metal

Apple Metal GPU backend plugin for [arrow-ml](https://crates.io/crates/arrow-ml).

This crate compiles to a `cdylib` (`libarrow_ml_backend_metal.dylib`) that
arrow-ml's runtime backend registry discovers and loads at runtime. There is
no Rust API surface — the cdylib is loaded via `dlopen`.

Data must be explicitly moved to a Metal device with `.to(Device::metal(0))`
before GPU kernels will run. See the main arrow-ml README for details.

## Installation

Add to your `Cargo.toml` alongside `arrow-ml`:

```toml
[dependencies]
arrow-ml = "0.1"
arrow-ml-backend-metal = "0.1"
```

When you `cargo build`, Cargo produces `libarrow_ml_backend_metal.dylib` in
your `target/{debug,release}/` directory. arrow-ml's runtime registry
searches there automatically.

For production deployments, copy the dylib next to your binary or set
`ARROW_ML_BACKEND_DIR` to a directory containing it.

## Discovery via manifest

A reference manifest ships at `manifests/metal.json` that you can drop into
`~/.arrow-ml/backends/` (or `$ARROW_ML_BACKEND_MANIFEST_DIR`) so the loader
can read backend metadata without `dlopen`ing the library first.

## Platform

macOS only. The crate fails to build on Linux/Windows because it depends on
Apple's `metal` framework.

## License

Apache-2.0
