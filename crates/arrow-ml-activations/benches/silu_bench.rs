use arrow::array::Float32Array;
use arrow_ml_activations::silu::silu;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_silu_f32(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_f32");
    for &size in &[256, 1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(BenchmarkId::new("arrow_silu", size), &size, |bench, _| {
            bench.iter(|| silu(&array))
        });

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_silu_f32(&data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_silu);
criterion_main!(benches);
