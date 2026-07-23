use std::hint::black_box;
use std::time::Duration;

use bit_readers::BitReader;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use decv_h264::{CoeffTokenContext, decode_coeff_token, decode_residual_block};

const BLOCK_COUNT: usize = 64 * 1024;

fn repeated_bits(code: &str, count: usize) -> Vec<u8> {
    let bit_len = code.len() * count;
    let mut bytes = vec![0u8; bit_len.div_ceil(8)];
    for block in 0..count {
        for (offset, bit) in code.bytes().enumerate() {
            if bit == b'1' {
                let position = block * code.len() + offset;
                bytes[position / 8] |= 1 << (7 - position % 8);
            }
        }
    }
    bytes
}

fn decode_tokens(data: &[u8], context: CoeffTokenContext, count: usize) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;
    for _ in 0..count {
        let token = decode_coeff_token(&mut reader, context, 16).unwrap();
        checksum = checksum
            .wrapping_add(u64::from(token.total_coeff))
            .wrapping_add(u64::from(token.trailing_ones));
    }
    checksum
}

fn coeff_tokens(criterion: &mut Criterion) {
    let cases = [
        ("nc_0_empty", "1", CoeffTokenContext::NeighborTotal(0)),
        ("nc_2_total_3", "0101", CoeffTokenContext::NeighborTotal(2)),
        ("nc_4_total_7", "1000", CoeffTokenContext::NeighborTotal(4)),
        ("chroma_dc_total_2", "001", CoeffTokenContext::ChromaDc420),
    ];
    let mut group = criterion.benchmark_group("coeff_token");
    group.throughput(Throughput::Elements(BLOCK_COUNT as u64));

    for (name, bits, context) in cases {
        let data = repeated_bits(bits, BLOCK_COUNT);
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |bencher, data| {
            bencher.iter(|| {
                black_box(decode_tokens(
                    black_box(data),
                    black_box(context),
                    BLOCK_COUNT,
                ))
            });
        });
    }
    group.finish();
}

fn decode_blocks(data: &[u8], context: CoeffTokenContext, max_num_coeff: u8, count: usize) -> i64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0i64;
    for _ in 0..count {
        let block = decode_residual_block(&mut reader, context, max_num_coeff).unwrap();
        checksum = checksum
            .wrapping_add(i64::from(block.total_coeff))
            .wrapping_add(
                block
                    .coefficients
                    .iter()
                    .map(|&value| i64::from(value))
                    .sum(),
            );
    }
    checksum
}

fn residual_blocks(criterion: &mut Criterion) {
    let cases = [
        (
            "single_level",
            "00010111",
            CoeffTokenContext::NeighborTotal(0),
            16,
        ),
        (
            "sparse_three",
            "00011010110011",
            CoeffTokenContext::NeighborTotal(0),
            16,
        ),
        ("chroma_dc", "00101011", CoeffTokenContext::ChromaDc420, 4),
    ];
    let mut group = criterion.benchmark_group("residual_block");
    group.throughput(Throughput::Elements(BLOCK_COUNT as u64));

    for (name, bits, context, max_num_coeff) in cases {
        let data = repeated_bits(bits, BLOCK_COUNT);
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |bencher, data| {
            bencher.iter(|| {
                black_box(decode_blocks(
                    black_box(data),
                    black_box(context),
                    black_box(max_num_coeff),
                    BLOCK_COUNT,
                ))
            });
        });
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
    targets = coeff_tokens, residual_blocks
}
criterion_main!(benches);
