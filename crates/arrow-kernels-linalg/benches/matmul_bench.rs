use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::tensor::Tensor;
use arrow_kernels_linalg::matmul::matmul;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_f32_tensor(rows: usize, cols: usize) -> Tensor<'static, Float32Type> {
    let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 100) as f32) * 0.01).collect();
    let buffer = Buffer::from(data);
    Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
}

fn make_f64_tensor(rows: usize, cols: usize) -> Tensor<'static, Float64Type> {
    let data: Vec<f64> = (0..rows * cols).map(|i| ((i % 100) as f64) * 0.01).collect();
    let buffer = Buffer::from(ScalarBuffer::<f64>::from(data).into_inner());
    Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
}

/// Naive triple-loop matmul for baseline comparison.
fn naive_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn bench_simd_vs_naive_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_naive_f32");
    for &size in &[32, 64, 128, 256, 512] {
        let a_tensor = make_f32_tensor(size, size);
        let b_tensor = make_f32_tensor(size, size);

        let a_data: Vec<f32> = (0..size * size).map(|i| ((i % 100) as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i % 100) as f32) * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| matmul(&a_tensor, &b_tensor).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, &s| {
            bench.iter(|| naive_matmul_f32(&a_data, &b_data, s, s, s))
        });
    }
    group.finish();
}

fn bench_matmul_f32_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32_square");
    for &size in &[16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let a = make_f32_tensor(size, size);
        let b = make_f32_tensor(size, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| matmul(&a, &b).unwrap())
        });
    }
    group.finish();
}

fn bench_matmul_f64_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f64_square");
    for &size in &[16, 32, 64, 128, 256, 512] {
        let a = make_f64_tensor(size, size);
        let b = make_f64_tensor(size, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| matmul(&a, &b).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_vs_naive_f32,
    bench_matmul_f32_square,
    bench_matmul_f64_square,
);

criterion_main!(benches);
