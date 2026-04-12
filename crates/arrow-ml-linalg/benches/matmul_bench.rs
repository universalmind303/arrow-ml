use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_core::tensor::Tensor;
use arrow_ml_linalg::matmul::matmul;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

fn make_f32_tensor(rows: usize, cols: usize) -> ArrowTensor<'static, Float32Type> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 100) as f32) * 0.01)
        .collect();
    let buffer = Buffer::from(data);
    ArrowTensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
}

fn make_f64_tensor(rows: usize, cols: usize) -> ArrowTensor<'static, Float64Type> {
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i % 100) as f64) * 0.01)
        .collect();
    let buffer = ScalarBuffer::<f64>::from(data).into_inner();
    ArrowTensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
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
        let a_tensor: Tensor = make_f32_tensor(size, size).into();
        let b_tensor: Tensor = make_f32_tensor(size, size).into();

        let a_data: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();

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
        let a: Tensor = make_f32_tensor(size, size).into();
        let b: Tensor = make_f32_tensor(size, size).into();

        let a_data: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();
        let a_nd = Array2::from_shape_vec((size, size), a_data.clone()).unwrap();
        let b_nd = Array2::from_shape_vec((size, size), b_data.clone()).unwrap();

        group.bench_with_input(BenchmarkId::new("arrow_ml", size), &size, |bench, _| {
            bench.iter(|| matmul(&a, &b).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| a_nd.dot(&b_nd))
        });

        group.bench_with_input(
            BenchmarkId::new("matrixmultiply", size),
            &size,
            |bench, &s| {
                bench.iter(|| {
                    let mut c = vec![0.0f32; s * s];
                    unsafe {
                        matrixmultiply::sgemm(
                            s,
                            s,
                            s,
                            1.0,
                            a_data.as_ptr(),
                            s as isize,
                            1,
                            b_data.as_ptr(),
                            s as isize,
                            1,
                            0.0,
                            c.as_mut_ptr(),
                            s as isize,
                            1,
                        );
                    }
                    c
                })
            },
        );
    }
    group.finish();
}

fn bench_matmul_f64_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f64_square");
    for &size in &[16, 32, 64, 128, 256, 512] {
        let a: Tensor = make_f64_tensor(size, size).into();
        let b: Tensor = make_f64_tensor(size, size).into();

        let a_data: Vec<f64> = (0..size * size)
            .map(|i| ((i % 100) as f64) * 0.01)
            .collect();
        let b_data: Vec<f64> = (0..size * size)
            .map(|i| ((i % 100) as f64) * 0.01)
            .collect();
        let a_nd = Array2::from_shape_vec((size, size), a_data.clone()).unwrap();
        let b_nd = Array2::from_shape_vec((size, size), b_data.clone()).unwrap();

        group.bench_with_input(BenchmarkId::new("arrow_ml", size), &size, |bench, _| {
            bench.iter(|| matmul(&a, &b).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| a_nd.dot(&b_nd))
        });

        group.bench_with_input(
            BenchmarkId::new("matrixmultiply", size),
            &size,
            |bench, &s| {
                bench.iter(|| {
                    let mut c = vec![0.0f64; s * s];
                    unsafe {
                        matrixmultiply::dgemm(
                            s,
                            s,
                            s,
                            1.0,
                            a_data.as_ptr(),
                            s as isize,
                            1,
                            b_data.as_ptr(),
                            s as isize,
                            1,
                            0.0,
                            c.as_mut_ptr(),
                            s as isize,
                            1,
                        );
                    }
                    c
                })
            },
        );
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
