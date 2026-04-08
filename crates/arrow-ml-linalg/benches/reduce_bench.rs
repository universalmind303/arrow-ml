use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::tensor::Tensor;
use arrow_ml_linalg::reduce::{reduce_max, reduce_mean, reduce_min, reduce_sum};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn make_f32_tensor(shape: Vec<usize>) -> Tensor<'static, Float32Type> {
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(|i| (i % 100) as f32 * 0.01).collect();
    let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
    Tensor::new_row_major(buffer, Some(shape), None).unwrap()
}

fn make_f64_tensor(shape: Vec<usize>) -> Tensor<'static, Float64Type> {
    let total: usize = shape.iter().product();
    let data: Vec<f64> = (0..total).map(|i| (i % 100) as f64 * 0.01).collect();
    let buffer = Buffer::from(ScalarBuffer::<f64>::from(data).into_inner());
    Tensor::new_row_major(buffer, Some(shape), None).unwrap()
}

// ---------------------------------------------------------------------------
// Naive baselines (scalar, no SIMD) used for comparison
// ---------------------------------------------------------------------------

fn naive_reduce_sum_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| data[r * cols..(r + 1) * cols].iter().sum())
        .collect()
}

fn naive_reduce_max_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            data[r * cols..(r + 1) * cols]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

fn naive_reduce_mean_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let s: f32 = data[r * cols..(r + 1) * cols].iter().sum();
            s / cols as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks: SIMD vs naive for reduce_sum over last axis (f32)
// ---------------------------------------------------------------------------

fn bench_reduce_sum_simd_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_sum_last_axis_simd_vs_naive");
    for &hidden in &[64usize, 256, 1024, 4096] {
        let rows = 128;
        let tensor = make_f32_tensor(vec![rows, hidden]);
        let raw: Vec<f32> = (0..rows * hidden)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("simd", hidden), &hidden, |b, _| {
            b.iter(|| reduce_sum::<Float32Type>(&tensor, &[-1], false).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("naive", hidden), &hidden, |b, _| {
            b.iter(|| naive_reduce_sum_f32(&raw, rows, hidden))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: SIMD vs naive for reduce_max over last axis (f32)
// ---------------------------------------------------------------------------

fn bench_reduce_max_simd_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_max_last_axis_simd_vs_naive");
    for &hidden in &[64usize, 256, 1024, 4096] {
        let rows = 128;
        let tensor = make_f32_tensor(vec![rows, hidden]);
        let raw: Vec<f32> = (0..rows * hidden)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("simd", hidden), &hidden, |b, _| {
            b.iter(|| reduce_max::<Float32Type>(&tensor, &[-1], false).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("naive", hidden), &hidden, |b, _| {
            b.iter(|| naive_reduce_max_f32(&raw, rows, hidden))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: SIMD vs naive for reduce_mean over last axis (f32)
// ---------------------------------------------------------------------------

fn bench_reduce_mean_simd_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_mean_last_axis_simd_vs_naive");
    for &hidden in &[64usize, 256, 1024, 4096] {
        let rows = 128;
        let tensor = make_f32_tensor(vec![rows, hidden]);
        let raw: Vec<f32> = (0..rows * hidden)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("simd", hidden), &hidden, |b, _| {
            b.iter(|| reduce_mean::<Float32Type>(&tensor, &[-1], false).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("naive", hidden), &hidden, |b, _| {
            b.iter(|| naive_reduce_mean_f32(&raw, rows, hidden))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: reduce_min – axis-0 (non-contiguous) vs axis-1 (contiguous)
// ---------------------------------------------------------------------------

fn bench_reduce_min_axis_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_min_axis_shapes");
    for &size in &[128usize, 512, 2048] {
        let tensor = make_f32_tensor(vec![size, size]);

        group.bench_with_input(
            BenchmarkId::new("axis0_noncontiguous", size),
            &size,
            |b, _| b.iter(|| reduce_min::<Float32Type>(&tensor, &[0], false).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("axis1_contiguous", size),
            &size,
            |b, _| b.iter(|| reduce_min::<Float32Type>(&tensor, &[1], false).unwrap()),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: f64 reduce_sum
// ---------------------------------------------------------------------------

fn bench_reduce_sum_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_sum_f64");
    for &hidden in &[64usize, 256, 1024] {
        let rows = 64;
        let tensor = make_f64_tensor(vec![rows, hidden]);
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, _| {
            b.iter(|| reduce_sum::<Float64Type>(&tensor, &[-1], false).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_reduce_sum_simd_vs_naive,
    bench_reduce_max_simd_vs_naive,
    bench_reduce_mean_simd_vs_naive,
    bench_reduce_min_axis_shapes,
    bench_reduce_sum_f64,
);
criterion_main!(benches);
