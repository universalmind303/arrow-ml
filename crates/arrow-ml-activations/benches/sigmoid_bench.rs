use arrow::array::Float32Array;
use arrow_ml_activations::sigmoid::sigmoid;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_sigmoid_f32(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid_f32");
    for &size in &[256, 1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(
            BenchmarkId::new("arrow_sigmoid", size),
            &size,
            |bench, _| bench.iter(|| sigmoid(&array)),
        );

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_sigmoid_f32(&data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sigmoid);
criterion_main!(benches);
