use arrow::array::Float32Array;
use arrow_ml_activations::softmax::softmax;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_softmax_f32(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.into_iter().map(|e| e / sum).collect()
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f32");
    for &size in &[256, 1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(
            BenchmarkId::new("arrow_softmax", size),
            &size,
            |bench, _| bench.iter(|| softmax(&array)),
        );

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_softmax_f32(&data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_softmax);
criterion_main!(benches);
