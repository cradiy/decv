use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use decv_core::Size;
use decv_h264::{MotionVector, ResolvedPPartition, Yuv420Picture};

fn partition(motion_vector: MotionVector) -> ResolvedPPartition {
    ResolvedPPartition {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
        reference_index: 0,
        motion_vector,
    }
}

fn inter_prediction(criterion: &mut Criterion) {
    let picture = Yuv420Picture::new(Size::new(1920, 1088)).unwrap();
    let cases = [
        (
            "integer_interior",
            60,
            30,
            partition(MotionVector { x: 0, y: 0 }),
        ),
        (
            "fractional_interior",
            60,
            30,
            partition(MotionVector { x: 3, y: 3 }),
        ),
        (
            "fractional_clipped_edge",
            0,
            0,
            partition(MotionVector { x: -3, y: -3 }),
        ),
    ];

    let mut group = criterion.benchmark_group("inter_prediction_16x16");
    group.throughput(Throughput::Elements(1));
    for (name, macroblock_x, macroblock_y, partition) in cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(macroblock_x, macroblock_y, partition),
            |bencher, &(macroblock_x, macroblock_y, partition)| {
                bencher.iter(|| {
                    black_box(
                        black_box(&picture)
                            .predict_inter_420(
                                black_box(macroblock_x),
                                black_box(macroblock_y),
                                black_box(partition),
                            )
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

fn configuration() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = configuration();
    targets = inter_prediction
}
criterion_main!(benches);
