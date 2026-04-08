use arrow::array::Float32Array;
use arrow_ml_activations::gelu::gelu;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_gelu_f32(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_f32");
    for &size in &[256, 1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(BenchmarkId::new("arrow_gelu", size), &size, |bench, _| {
            bench.iter(|| gelu(&array))
        });

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_gelu_f32(&data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gelu);
criterion_main!(benches);
