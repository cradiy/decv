use std::hint::black_box;
use std::time::Duration;

use bit_readers::BitReader;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const INPUT_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy)]
struct NaiveBitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> NaiveBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn remaining_bits(&self) -> usize {
        self.data.len() * 8 - self.position
    }

    fn read_bits(&mut self, count: u32) -> Option<u32> {
        if count > 32 || count as usize > self.remaining_bits() {
            return None;
        }

        let mut value = 0;
        for _ in 0..count {
            let byte = self.data[self.position / 8];
            let bit = (byte >> (7 - self.position % 8)) & 1;
            value = (value << 1) | bit as u32;
            self.position += 1;
        }
        Some(value)
    }

    fn peek_bits(&self, count: u32) -> Option<u32> {
        let mut copy = *self;
        copy.read_bits(count)
    }

    fn skip_bits(&mut self, count: usize) -> bool {
        if count > self.remaining_bits() {
            return false;
        }
        self.position += count;
        true
    }

    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0;

        while self.read_bits(1)? == 0 {
            leading_zeros += 1;
            if leading_zeros > 32 {
                return None;
            }
        }

        let suffix = self.read_bits(leading_zeros)?;
        let value = ((1u64 << leading_zeros) - 1) + suffix as u64;
        u32::try_from(value).ok()
    }
}

fn input() -> Vec<u8> {
    (0..INPUT_BYTES)
        .map(|index| {
            let index = index as u32;
            index
                .wrapping_mul(73)
                .wrapping_add(index.rotate_left(7))
                .wrapping_add(19) as u8
        })
        .collect()
}

fn ue_input() -> Vec<u8> {
    const VALUES: [u32; 20] = [
        0, 0, 1, 0, 2, 1, 3, 0, 4, 2, 7, 1, 15, 5, 31, 0, 2, 255, 3, 65_535,
    ];

    let mut bits = Vec::with_capacity(INPUT_BYTES * 8);
    let mut index = 0;

    while bits.len() < INPUT_BYTES * 8 {
        let code_num = VALUES[index % VALUES.len()] as u64 + 1;
        let width = 64 - code_num.leading_zeros();

        bits.extend(std::iter::repeat_n(0, width as usize - 1));
        for shift in (0..width).rev() {
            bits.push(((code_num >> shift) & 1) as u8);
        }
        index += 1;
    }

    let mut bytes = vec![0; bits.len().div_ceil(8)];
    for (position, bit) in bits.into_iter().enumerate() {
        bytes[position / 8] |= bit << (7 - position % 8);
    }
    bytes
}

fn read_fixed_reservoir(data: &[u8], width: u32) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    while let Some(value) = reader.read_bits(width) {
        checksum = checksum.wrapping_add(value as u64);
    }

    checksum
}

fn read_fixed_naive(data: &[u8], width: u32) -> u64 {
    let mut reader = NaiveBitReader::new(data);
    let mut checksum = 0u64;

    while let Some(value) = reader.read_bits(width) {
        checksum = checksum.wrapping_add(value as u64);
    }

    checksum
}

fn read_fixed_const<const WIDTH: u32>(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    while let Some(value) = reader.read_bits_const::<WIDTH>() {
        checksum = checksum.wrapping_add(value as u64);
    }

    checksum
}

fn fixed_widths(criterion: &mut Criterion) {
    let data = input();
    let mut group = criterion.benchmark_group("fixed_width");
    group.throughput(Throughput::Bytes(data.len() as u64));

    for width in [1, 8, 17, 32] {
        group.bench_with_input(
            BenchmarkId::new("reservoir", width),
            &width,
            |bencher, &width| {
                bencher.iter(|| black_box(read_fixed_reservoir(black_box(&data), width)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("naive", width),
            &width,
            |bencher, &width| {
                bencher.iter(|| black_box(read_fixed_naive(black_box(&data), width)));
            },
        );
    }

    group.bench_function(BenchmarkId::new("const", 1), |bencher| {
        bencher.iter(|| black_box(read_fixed_const::<1>(black_box(&data))));
    });
    group.bench_function(BenchmarkId::new("const", 8), |bencher| {
        bencher.iter(|| black_box(read_fixed_const::<8>(black_box(&data))));
    });
    group.bench_function(BenchmarkId::new("const", 17), |bencher| {
        bencher.iter(|| black_box(read_fixed_const::<17>(black_box(&data))));
    });
    group.bench_function(BenchmarkId::new("const", 32), |bencher| {
        bencher.iter(|| black_box(read_fixed_const::<32>(black_box(&data))));
    });

    group.finish();
}

fn read_single_bits_reservoir(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    while let Some(bit) = reader.read_bit() {
        checksum = checksum.wrapping_add(bit as u64);
    }

    checksum
}

fn read_single_bits_naive(data: &[u8]) -> u64 {
    let mut reader = NaiveBitReader::new(data);
    let mut checksum = 0u64;

    while let Some(bit) = reader.read_bits(1) {
        checksum = checksum.wrapping_add(bit as u64);
    }

    checksum
}

fn single_bits(criterion: &mut Criterion) {
    let data = input();
    let mut group = criterion.benchmark_group("single_bits");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("reservoir", |bencher| {
        bencher.iter(|| black_box(read_single_bits_reservoir(black_box(&data))));
    });
    group.bench_function("naive", |bencher| {
        bencher.iter(|| black_box(read_single_bits_naive(black_box(&data))));
    });

    group.finish();
}

const VIDEO_FIELD_WIDTHS: [u32; 16] = [1, 2, 5, 1, 8, 16, 3, 7, 24, 1, 12, 4, 32, 6, 2, 9];

fn read_mixed_reservoir(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;
    let mut field = 0;

    loop {
        let width = VIDEO_FIELD_WIDTHS[field % VIDEO_FIELD_WIDTHS.len()];
        let Some(value) = reader.read_bits(width) else {
            break;
        };
        checksum = checksum.wrapping_add(value as u64);
        field += 1;
    }

    checksum
}

fn read_mixed_naive(data: &[u8]) -> u64 {
    let mut reader = NaiveBitReader::new(data);
    let mut checksum = 0u64;
    let mut field = 0;

    loop {
        let width = VIDEO_FIELD_WIDTHS[field % VIDEO_FIELD_WIDTHS.len()];
        let Some(value) = reader.read_bits(width) else {
            break;
        };
        checksum = checksum.wrapping_add(value as u64);
        field += 1;
    }

    checksum
}

fn read_mixed_const(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    macro_rules! read {
        ($width:literal) => {
            let Some(value) = reader.read_bits_const::<$width>() else {
                return checksum;
            };
            checksum = checksum.wrapping_add(value as u64);
        };
    }

    loop {
        read!(1);
        read!(2);
        read!(5);
        read!(1);
        read!(8);
        read!(16);
        read!(3);
        read!(7);
        read!(24);
        read!(1);
        read!(12);
        read!(4);
        read!(32);
        read!(6);
        read!(2);
        read!(9);
    }
}

fn mixed_video_fields(criterion: &mut Criterion) {
    let data = input();
    let mut group = criterion.benchmark_group("mixed_video_fields");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("reservoir", |bencher| {
        bencher.iter(|| black_box(read_mixed_reservoir(black_box(&data))));
    });
    group.bench_function("naive", |bencher| {
        bencher.iter(|| black_box(read_mixed_naive(black_box(&data))));
    });
    group.bench_function("const", |bencher| {
        bencher.iter(|| black_box(read_mixed_const(black_box(&data))));
    });
    group.finish();
}

fn read_ue_reservoir(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    while let Some(value) = reader.read_ue() {
        checksum = checksum.wrapping_add(value as u64);
    }

    checksum
}

fn read_ue_naive(data: &[u8]) -> u64 {
    let mut reader = NaiveBitReader::new(data);
    let mut checksum = 0u64;

    while let Some(value) = reader.read_ue() {
        checksum = checksum.wrapping_add(value as u64);
    }

    checksum
}

fn exponential_golomb(criterion: &mut Criterion) {
    let data = ue_input();
    let mut group = criterion.benchmark_group("exponential_golomb");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("reservoir", |bencher| {
        bencher.iter(|| black_box(read_ue_reservoir(black_box(&data))));
    });
    group.bench_function("naive", |bencher| {
        bencher.iter(|| black_box(read_ue_naive(black_box(&data))));
    });

    group.finish();
}

fn peek_skip_reservoir(data: &[u8]) -> u64 {
    let mut reader = BitReader::new(data);
    let mut checksum = 0u64;

    while reader.remaining_bits() >= 16 {
        checksum = checksum.wrapping_add(reader.peek_bits(16).unwrap() as u64);
        assert!(reader.skip_bits(5));
    }

    checksum
}

fn peek_skip_naive(data: &[u8]) -> u64 {
    let mut reader = NaiveBitReader::new(data);
    let mut checksum = 0u64;

    while reader.remaining_bits() >= 16 {
        checksum = checksum.wrapping_add(reader.peek_bits(16).unwrap() as u64);
        assert!(reader.skip_bits(5));
    }

    checksum
}

fn peek_and_skip(criterion: &mut Criterion) {
    let data = input();
    let mut group = criterion.benchmark_group("peek_and_skip");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("reservoir", |bencher| {
        bencher.iter(|| black_box(peek_skip_reservoir(black_box(&data))));
    });
    group.bench_function("naive", |bencher| {
        bencher.iter(|| black_box(peek_skip_naive(black_box(&data))));
    });

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
    targets =
        fixed_widths,
        single_bits,
        mixed_video_fields,
        exponential_golomb,
        peek_and_skip
}
criterion_main!(benches);
