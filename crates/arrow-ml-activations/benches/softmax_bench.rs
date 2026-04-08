use arrow::array::Float32Array;
use arrow_ml_activations::softmax::softmax;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_softmax_f32(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }

    let mut sum = 0.0f32;
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }

    let inv_sum = 1.0 / sum;
    for i in 0..n {
        output[i] *= inv_sum;
    }

    output
}

fn bench_softmax_naive_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_naive_vs_simd");

    for &size in &[64, 256, 1024, 4096, 16384] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, _| {
            b.iter(|| softmax(black_box(&array)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| naive_softmax_f32(black_box(&data)))
        });
    }

    group.finish();
}

fn bench_softmax_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_sizes");

    for &size in &[
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ] {
        let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) * 0.01).collect();
        let array = Float32Array::from(data);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| softmax(black_box(&array)).unwrap())
        });
    }

    group.finish();
}

fn bench_softmax_threshold_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_threshold");

    // Test around the SIMD threshold (1024)
    for &size in &[512, 768, 1024, 1280, 1536, 2048] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let array = Float32Array::from(data);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| softmax(black_box(&array)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_softmax_naive_vs_simd,
    bench_softmax_sizes,
    bench_softmax_threshold_crossover,
);

criterion_main!(benches);
