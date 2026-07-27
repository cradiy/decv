// VP9's inverse transforms use fixed-point cosine constants and normative
// rounding between butterfly stages. Keeping the scalar kernels here makes
// their arithmetic independently testable and gives SIMD implementations a
// stable interface to replace later.

const COSPI: [i32; 32] = [
    0, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623, 13160,
    12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981, 3196,
    2404, 1606, 804,
];
const SINPI_1_9: i32 = 5283;
const SINPI_2_9: i32 = 9929;
const SINPI_3_9: i32 = 13377;
const SINPI_4_9: i32 = 15212;

#[inline(always)]
fn cospi(index: usize) -> i32 {
    COSPI[index]
}

#[inline(always)]
fn round_shift(value: i64) -> i32 {
    ((value + (1 << 13)) >> 14) as i32
}

pub(crate) fn inverse_dct(input: &[i32; 32], output: &mut [i32; 32], size: usize) {
    match size {
        4 => idct4(input, output),
        8 => idct8(input, output),
        16 => idct16(input, output),
        32 => idct32(input, output),
        _ => unreachable!("VP9 transform dimensions are powers of two from 4 to 32"),
    }
}

pub(crate) fn inverse_adst(input: &[i32; 32], output: &mut [i32; 32], size: usize) {
    match size {
        4 => iadst4(input, output),
        8 => iadst8(input, output),
        16 => iadst16(input, output),
        32 => unreachable!("VP9 does not define a 32-point ADST"),
        _ => unreachable!("VP9 transform dimensions are powers of two from 4 to 16"),
    }
}

fn idct4(input: &[i32; 32], output: &mut [i32; 32]) {
    let input = input.map(|value| i32::from(value as i16));
    let mut step = [0i32; 4];
    step[0] = round_shift(i64::from(input[0] + input[2]) * i64::from(cospi(16)));
    step[1] = round_shift(i64::from(input[0] - input[2]) * i64::from(cospi(16)));
    step[2] = round_shift(
        i64::from(input[1]) * i64::from(cospi(24)) - i64::from(input[3]) * i64::from(cospi(8)),
    );
    step[3] = round_shift(
        i64::from(input[1]) * i64::from(cospi(8)) + i64::from(input[3]) * i64::from(cospi(24)),
    );
    output[0] = step[0] + step[3];
    output[1] = step[1] + step[2];
    output[2] = step[1] - step[2];
    output[3] = step[0] - step[3];
}

fn iadst4(input: &[i32; 32], output: &mut [i32; 32]) {
    let [x0, x1, x2, x3] = input[..4].try_into().unwrap();
    if x0 | x1 | x2 | x3 == 0 {
        output[..4].fill(0);
        return;
    }
    let s0 = i64::from(SINPI_1_9) * i64::from(x0);
    let s1 = i64::from(SINPI_2_9) * i64::from(x0);
    let s2 = i64::from(SINPI_3_9) * i64::from(x1);
    let s3 = i64::from(SINPI_4_9) * i64::from(x2);
    let s4 = i64::from(SINPI_1_9) * i64::from(x2);
    let s5 = i64::from(SINPI_2_9) * i64::from(x3);
    let s6 = i64::from(SINPI_4_9) * i64::from(x3);
    let s7 = x0 - x2 + x3;

    let a = s0 + s3 + s5;
    let b = s1 - s4 - s6;
    let c = s2;
    let d = i64::from(SINPI_3_9) * i64::from(s7);
    output[0] = round_shift(a + c);
    output[1] = round_shift(b + c);
    output[2] = round_shift(d);
    output[3] = round_shift(a + b - c);
}

fn idct8(input: &[i32; 32], output: &mut [i32; 32]) {
    let input = input.map(|value| i32::from(value as i16));
    let mut step1 = [0i32; 8];
    let mut step2 = [0i32; 8];

    step1[0] = input[0];
    step1[1] = input[2];
    step1[2] = input[4];
    step1[3] = input[6];
    step1[4] = round_shift(
        i64::from(input[1]) * i64::from(cospi(28)) - i64::from(input[7]) * i64::from(cospi(4)),
    );
    step1[7] = round_shift(
        i64::from(input[1]) * i64::from(cospi(4)) + i64::from(input[7]) * i64::from(cospi(28)),
    );
    step1[5] = round_shift(
        i64::from(input[5]) * i64::from(cospi(12)) - i64::from(input[3]) * i64::from(cospi(20)),
    );
    step1[6] = round_shift(
        i64::from(input[5]) * i64::from(cospi(20)) + i64::from(input[3]) * i64::from(cospi(12)),
    );

    step2[0] = round_shift(i64::from(step1[0] + step1[2]) * i64::from(cospi(16)));
    step2[1] = round_shift(i64::from(step1[0] - step1[2]) * i64::from(cospi(16)));
    step2[2] = round_shift(
        i64::from(step1[1]) * i64::from(cospi(24)) - i64::from(step1[3]) * i64::from(cospi(8)),
    );
    step2[3] = round_shift(
        i64::from(step1[1]) * i64::from(cospi(8)) + i64::from(step1[3]) * i64::from(cospi(24)),
    );
    step2[4] = step1[4] + step1[5];
    step2[5] = step1[4] - step1[5];
    step2[6] = -step1[6] + step1[7];
    step2[7] = step1[6] + step1[7];

    step1[0] = step2[0] + step2[3];
    step1[1] = step2[1] + step2[2];
    step1[2] = step2[1] - step2[2];
    step1[3] = step2[0] - step2[3];
    step1[4] = step2[4];
    step1[5] = round_shift(i64::from(step2[6] - step2[5]) * i64::from(cospi(16)));
    step1[6] = round_shift(i64::from(step2[5] + step2[6]) * i64::from(cospi(16)));
    step1[7] = step2[7];

    output[0] = step1[0] + step1[7];
    output[1] = step1[1] + step1[6];
    output[2] = step1[2] + step1[5];
    output[3] = step1[3] + step1[4];
    output[4] = step1[3] - step1[4];
    output[5] = step1[2] - step1[5];
    output[6] = step1[1] - step1[6];
    output[7] = step1[0] - step1[7];
}

fn iadst8(input: &[i32; 32], output: &mut [i32; 32]) {
    let mut x = [
        input[7], input[0], input[5], input[2], input[3], input[4], input[1], input[6],
    ];
    if x.iter().all(|&value| value == 0) {
        output[..8].fill(0);
        return;
    }
    let mut s = [0i64; 8];
    s[0] = i64::from(cospi(2)) * i64::from(x[0]) + i64::from(cospi(30)) * i64::from(x[1]);
    s[1] = i64::from(cospi(30)) * i64::from(x[0]) - i64::from(cospi(2)) * i64::from(x[1]);
    s[2] = i64::from(cospi(10)) * i64::from(x[2]) + i64::from(cospi(22)) * i64::from(x[3]);
    s[3] = i64::from(cospi(22)) * i64::from(x[2]) - i64::from(cospi(10)) * i64::from(x[3]);
    s[4] = i64::from(cospi(18)) * i64::from(x[4]) + i64::from(cospi(14)) * i64::from(x[5]);
    s[5] = i64::from(cospi(14)) * i64::from(x[4]) - i64::from(cospi(18)) * i64::from(x[5]);
    s[6] = i64::from(cospi(26)) * i64::from(x[6]) + i64::from(cospi(6)) * i64::from(x[7]);
    s[7] = i64::from(cospi(6)) * i64::from(x[6]) - i64::from(cospi(26)) * i64::from(x[7]);
    for index in 0..4 {
        x[index] = round_shift(s[index] + s[index + 4]);
        x[index + 4] = round_shift(s[index] - s[index + 4]);
    }

    s[0] = i64::from(x[0]);
    s[1] = i64::from(x[1]);
    s[2] = i64::from(x[2]);
    s[3] = i64::from(x[3]);
    s[4] = i64::from(cospi(8)) * i64::from(x[4]) + i64::from(cospi(24)) * i64::from(x[5]);
    s[5] = i64::from(cospi(24)) * i64::from(x[4]) - i64::from(cospi(8)) * i64::from(x[5]);
    s[6] = -i64::from(cospi(24)) * i64::from(x[6]) + i64::from(cospi(8)) * i64::from(x[7]);
    s[7] = i64::from(cospi(8)) * i64::from(x[6]) + i64::from(cospi(24)) * i64::from(x[7]);
    x[0] = (s[0] + s[2]) as i32;
    x[1] = (s[1] + s[3]) as i32;
    x[2] = (s[0] - s[2]) as i32;
    x[3] = (s[1] - s[3]) as i32;
    x[4] = round_shift(s[4] + s[6]);
    x[5] = round_shift(s[5] + s[7]);
    x[6] = round_shift(s[4] - s[6]);
    x[7] = round_shift(s[5] - s[7]);

    let x2 = round_shift(i64::from(cospi(16)) * i64::from(x[2] + x[3]));
    let x3 = round_shift(i64::from(cospi(16)) * i64::from(x[2] - x[3]));
    let x6 = round_shift(i64::from(cospi(16)) * i64::from(x[6] + x[7]));
    let x7 = round_shift(i64::from(cospi(16)) * i64::from(x[6] - x[7]));
    output[..8].copy_from_slice(&[x[0], -x[4], x6, -x2, x3, -x7, x[5], -x[1]]);
}

fn idct16(input: &[i32; 32], output: &mut [i32; 32]) {
    let input = input.map(|value| i32::from(value as i16));
    let mut step1 = [0i32; 16];
    let mut step2 = [0i32; 16];

    for (target, source) in [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        .into_iter()
        .enumerate()
    {
        step1[target] = input[source];
    }
    step2[..8].copy_from_slice(&step1[..8]);
    for (low, high, first, second) in [
        (8, 15, 30, 2),
        (9, 14, 14, 18),
        (10, 13, 22, 10),
        (11, 12, 6, 26),
    ] {
        step2[low] = round_shift(
            i64::from(step1[low]) * i64::from(cospi(first))
                - i64::from(step1[high]) * i64::from(cospi(second)),
        );
        step2[high] = round_shift(
            i64::from(step1[low]) * i64::from(cospi(second))
                + i64::from(step1[high]) * i64::from(cospi(first)),
        );
    }

    step1[..4].copy_from_slice(&step2[..4]);
    for (low, high, first, second) in [(4, 7, 28, 4), (5, 6, 12, 20)] {
        step1[low] = round_shift(
            i64::from(step2[low]) * i64::from(cospi(first))
                - i64::from(step2[high]) * i64::from(cospi(second)),
        );
        step1[high] = round_shift(
            i64::from(step2[low]) * i64::from(cospi(second))
                + i64::from(step2[high]) * i64::from(cospi(first)),
        );
    }
    step1[8] = step2[8] + step2[9];
    step1[9] = step2[8] - step2[9];
    step1[10] = -step2[10] + step2[11];
    step1[11] = step2[10] + step2[11];
    step1[12] = step2[12] + step2[13];
    step1[13] = step2[12] - step2[13];
    step1[14] = -step2[14] + step2[15];
    step1[15] = step2[14] + step2[15];

    step2[0] = round_shift(i64::from(step1[0] + step1[1]) * i64::from(cospi(16)));
    step2[1] = round_shift(i64::from(step1[0] - step1[1]) * i64::from(cospi(16)));
    step2[2] = round_shift(
        i64::from(step1[2]) * i64::from(cospi(24)) - i64::from(step1[3]) * i64::from(cospi(8)),
    );
    step2[3] = round_shift(
        i64::from(step1[2]) * i64::from(cospi(8)) + i64::from(step1[3]) * i64::from(cospi(24)),
    );
    step2[4] = step1[4] + step1[5];
    step2[5] = step1[4] - step1[5];
    step2[6] = -step1[6] + step1[7];
    step2[7] = step1[6] + step1[7];
    step2[8] = step1[8];
    step2[15] = step1[15];
    step2[9] = round_shift(
        -i64::from(step1[9]) * i64::from(cospi(8)) + i64::from(step1[14]) * i64::from(cospi(24)),
    );
    step2[14] = round_shift(
        i64::from(step1[9]) * i64::from(cospi(24)) + i64::from(step1[14]) * i64::from(cospi(8)),
    );
    step2[10] = round_shift(
        -i64::from(step1[10]) * i64::from(cospi(24)) - i64::from(step1[13]) * i64::from(cospi(8)),
    );
    step2[13] = round_shift(
        -i64::from(step1[10]) * i64::from(cospi(8)) + i64::from(step1[13]) * i64::from(cospi(24)),
    );
    step2[11] = step1[11];
    step2[12] = step1[12];

    step1[0] = step2[0] + step2[3];
    step1[1] = step2[1] + step2[2];
    step1[2] = step2[1] - step2[2];
    step1[3] = step2[0] - step2[3];
    step1[4] = step2[4];
    step1[5] = round_shift(i64::from(step2[6] - step2[5]) * i64::from(cospi(16)));
    step1[6] = round_shift(i64::from(step2[5] + step2[6]) * i64::from(cospi(16)));
    step1[7] = step2[7];
    step1[8] = step2[8] + step2[11];
    step1[9] = step2[9] + step2[10];
    step1[10] = step2[9] - step2[10];
    step1[11] = step2[8] - step2[11];
    step1[12] = -step2[12] + step2[15];
    step1[13] = -step2[13] + step2[14];
    step1[14] = step2[13] + step2[14];
    step1[15] = step2[12] + step2[15];

    step2[0] = step1[0] + step1[7];
    step2[1] = step1[1] + step1[6];
    step2[2] = step1[2] + step1[5];
    step2[3] = step1[3] + step1[4];
    step2[4] = step1[3] - step1[4];
    step2[5] = step1[2] - step1[5];
    step2[6] = step1[1] - step1[6];
    step2[7] = step1[0] - step1[7];
    step2[8] = step1[8];
    step2[9] = step1[9];
    step2[10] = round_shift(i64::from(-step1[10] + step1[13]) * i64::from(cospi(16)));
    step2[13] = round_shift(i64::from(step1[10] + step1[13]) * i64::from(cospi(16)));
    step2[11] = round_shift(i64::from(-step1[11] + step1[12]) * i64::from(cospi(16)));
    step2[12] = round_shift(i64::from(step1[11] + step1[12]) * i64::from(cospi(16)));
    step2[14] = step1[14];
    step2[15] = step1[15];

    for index in 0..8 {
        output[index] = step2[index] + step2[15 - index];
        output[15 - index] = step2[index] - step2[15 - index];
    }
}

fn iadst16(input: &[i32; 32], output: &mut [i32; 32]) {
    let mut x = [
        input[15], input[0], input[13], input[2], input[11], input[4], input[9], input[6],
        input[7], input[8], input[5], input[10], input[3], input[12], input[1], input[14],
    ];
    if x.iter().all(|&value| value == 0) {
        output[..16].fill(0);
        return;
    }
    let mut s = [0i64; 16];
    for (pair, first, second) in [
        (0, 1, 31),
        (2, 5, 27),
        (4, 9, 23),
        (6, 13, 19),
        (8, 17, 15),
        (10, 21, 11),
        (12, 25, 7),
        (14, 29, 3),
    ] {
        s[pair] = i64::from(x[pair]) * i64::from(cospi(first))
            + i64::from(x[pair + 1]) * i64::from(cospi(second));
        s[pair + 1] = i64::from(x[pair]) * i64::from(cospi(second))
            - i64::from(x[pair + 1]) * i64::from(cospi(first));
    }
    for index in 0..8 {
        x[index] = round_shift(s[index] + s[index + 8]);
        x[index + 8] = round_shift(s[index] - s[index + 8]);
    }

    for index in 0..8 {
        s[index] = i64::from(x[index]);
    }
    s[8] = i64::from(x[8]) * i64::from(cospi(4)) + i64::from(x[9]) * i64::from(cospi(28));
    s[9] = i64::from(x[8]) * i64::from(cospi(28)) - i64::from(x[9]) * i64::from(cospi(4));
    s[10] = i64::from(x[10]) * i64::from(cospi(20)) + i64::from(x[11]) * i64::from(cospi(12));
    s[11] = i64::from(x[10]) * i64::from(cospi(12)) - i64::from(x[11]) * i64::from(cospi(20));
    s[12] = -i64::from(x[12]) * i64::from(cospi(28)) + i64::from(x[13]) * i64::from(cospi(4));
    s[13] = i64::from(x[12]) * i64::from(cospi(4)) + i64::from(x[13]) * i64::from(cospi(28));
    s[14] = -i64::from(x[14]) * i64::from(cospi(12)) + i64::from(x[15]) * i64::from(cospi(20));
    s[15] = i64::from(x[14]) * i64::from(cospi(20)) + i64::from(x[15]) * i64::from(cospi(12));
    for index in 0..4 {
        x[index] = (s[index] + s[index + 4]) as i32;
        x[index + 4] = (s[index] - s[index + 4]) as i32;
    }
    for index in 8..12 {
        x[index] = round_shift(s[index] + s[index + 4]);
        x[index + 4] = round_shift(s[index] - s[index + 4]);
    }

    for index in 0..4 {
        s[index] = i64::from(x[index]);
    }
    s[4] = i64::from(x[4]) * i64::from(cospi(8)) + i64::from(x[5]) * i64::from(cospi(24));
    s[5] = i64::from(x[4]) * i64::from(cospi(24)) - i64::from(x[5]) * i64::from(cospi(8));
    s[6] = -i64::from(x[6]) * i64::from(cospi(24)) + i64::from(x[7]) * i64::from(cospi(8));
    s[7] = i64::from(x[6]) * i64::from(cospi(8)) + i64::from(x[7]) * i64::from(cospi(24));
    for index in 8..12 {
        s[index] = i64::from(x[index]);
    }
    s[12] = i64::from(x[12]) * i64::from(cospi(8)) + i64::from(x[13]) * i64::from(cospi(24));
    s[13] = i64::from(x[12]) * i64::from(cospi(24)) - i64::from(x[13]) * i64::from(cospi(8));
    s[14] = -i64::from(x[14]) * i64::from(cospi(24)) + i64::from(x[15]) * i64::from(cospi(8));
    s[15] = i64::from(x[14]) * i64::from(cospi(8)) + i64::from(x[15]) * i64::from(cospi(24));
    for base in [0, 8] {
        x[base] = (s[base] + s[base + 2]) as i32;
        x[base + 1] = (s[base + 1] + s[base + 3]) as i32;
        x[base + 2] = (s[base] - s[base + 2]) as i32;
        x[base + 3] = (s[base + 1] - s[base + 3]) as i32;
        x[base + 4] = round_shift(s[base + 4] + s[base + 6]);
        x[base + 5] = round_shift(s[base + 5] + s[base + 7]);
        x[base + 6] = round_shift(s[base + 4] - s[base + 6]);
        x[base + 7] = round_shift(s[base + 5] - s[base + 7]);
    }

    for (left, right, first_sign, second_sign) in [
        (2, 3, -1, 1),
        (6, 7, 1, -1),
        (10, 11, 1, -1),
        (14, 15, -1, 1),
    ] {
        let first = first_sign * cospi(16) * (x[left] + x[right]);
        let second = second_sign * cospi(16) * (x[left] - x[right]);
        x[left] = round_shift(i64::from(first));
        x[right] = round_shift(i64::from(second));
    }
    output[..16].copy_from_slice(&[
        x[0], -x[8], x[12], -x[4], x[6], x[14], x[10], x[2], x[3], x[11], x[15], x[7], x[5],
        -x[13], x[9], -x[1],
    ]);
}

fn idct32(input: &[i32; 32], output: &mut [i32; 32]) {
    let input = input.map(|value| i32::from(value as i16));
    let mut step1 = [0i32; 32];
    let mut step2 = [0i32; 32];

    for (target, source) in [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30]
        .into_iter()
        .enumerate()
    {
        step1[target] = input[source];
    }
    for (target, low, high, first, second) in [
        (16, 1, 31, 31, 1),
        (17, 17, 15, 15, 17),
        (18, 9, 23, 23, 9),
        (19, 25, 7, 7, 25),
        (20, 5, 27, 27, 5),
        (21, 21, 11, 11, 21),
        (22, 13, 19, 19, 13),
        (23, 29, 3, 3, 29),
    ] {
        step1[target] = round_shift(
            i64::from(input[low]) * i64::from(cospi(first))
                - i64::from(input[high]) * i64::from(cospi(second)),
        );
        step1[47 - target] = round_shift(
            i64::from(input[low]) * i64::from(cospi(second))
                + i64::from(input[high]) * i64::from(cospi(first)),
        );
    }

    step2[..8].copy_from_slice(&step1[..8]);
    for (low, high, first, second) in [
        (8, 15, 30, 2),
        (9, 14, 14, 18),
        (10, 13, 22, 10),
        (11, 12, 6, 26),
    ] {
        step2[low] = round_shift(
            i64::from(step1[low]) * i64::from(cospi(first))
                - i64::from(step1[high]) * i64::from(cospi(second)),
        );
        step2[high] = round_shift(
            i64::from(step1[low]) * i64::from(cospi(second))
                + i64::from(step1[high]) * i64::from(cospi(first)),
        );
    }
    for base in (16..32).step_by(4) {
        step2[base] = step1[base] + step1[base + 1];
        step2[base + 1] = step1[base] - step1[base + 1];
        step2[base + 2] = -step1[base + 2] + step1[base + 3];
        step2[base + 3] = step1[base + 2] + step1[base + 3];
    }

    step1[..4].copy_from_slice(&step2[..4]);
    for (low, high, first, second) in [(4, 7, 28, 4), (5, 6, 12, 20)] {
        step1[low] = round_shift(
            i64::from(step2[low]) * i64::from(cospi(first))
                - i64::from(step2[high]) * i64::from(cospi(second)),
        );
        step1[high] = round_shift(
            i64::from(step2[low]) * i64::from(cospi(second))
                + i64::from(step2[high]) * i64::from(cospi(first)),
        );
    }
    step1[8] = step2[8] + step2[9];
    step1[9] = step2[8] - step2[9];
    step1[10] = -step2[10] + step2[11];
    step1[11] = step2[10] + step2[11];
    step1[12] = step2[12] + step2[13];
    step1[13] = step2[12] - step2[13];
    step1[14] = -step2[14] + step2[15];
    step1[15] = step2[14] + step2[15];
    step1[16] = step2[16];
    step1[31] = step2[31];
    step1[17] = round_shift(
        -i64::from(step2[17]) * i64::from(cospi(4)) + i64::from(step2[30]) * i64::from(cospi(28)),
    );
    step1[30] = round_shift(
        i64::from(step2[17]) * i64::from(cospi(28)) + i64::from(step2[30]) * i64::from(cospi(4)),
    );
    step1[18] = round_shift(
        -i64::from(step2[18]) * i64::from(cospi(28)) - i64::from(step2[29]) * i64::from(cospi(4)),
    );
    step1[29] = round_shift(
        -i64::from(step2[18]) * i64::from(cospi(4)) + i64::from(step2[29]) * i64::from(cospi(28)),
    );
    step1[19] = step2[19];
    step1[20] = step2[20];
    step1[21] = round_shift(
        -i64::from(step2[21]) * i64::from(cospi(20)) + i64::from(step2[26]) * i64::from(cospi(12)),
    );
    step1[26] = round_shift(
        i64::from(step2[21]) * i64::from(cospi(12)) + i64::from(step2[26]) * i64::from(cospi(20)),
    );
    step1[22] = round_shift(
        -i64::from(step2[22]) * i64::from(cospi(12)) - i64::from(step2[25]) * i64::from(cospi(20)),
    );
    step1[25] = round_shift(
        -i64::from(step2[22]) * i64::from(cospi(20)) + i64::from(step2[25]) * i64::from(cospi(12)),
    );
    for index in [23, 24, 27, 28] {
        step1[index] = step2[index];
    }

    step2[0] = round_shift(i64::from(step1[0] + step1[1]) * i64::from(cospi(16)));
    step2[1] = round_shift(i64::from(step1[0] - step1[1]) * i64::from(cospi(16)));
    step2[2] = round_shift(
        i64::from(step1[2]) * i64::from(cospi(24)) - i64::from(step1[3]) * i64::from(cospi(8)),
    );
    step2[3] = round_shift(
        i64::from(step1[2]) * i64::from(cospi(8)) + i64::from(step1[3]) * i64::from(cospi(24)),
    );
    step2[4] = step1[4] + step1[5];
    step2[5] = step1[4] - step1[5];
    step2[6] = -step1[6] + step1[7];
    step2[7] = step1[6] + step1[7];
    step2[8] = step1[8];
    step2[15] = step1[15];
    step2[9] = round_shift(
        -i64::from(step1[9]) * i64::from(cospi(8)) + i64::from(step1[14]) * i64::from(cospi(24)),
    );
    step2[14] = round_shift(
        i64::from(step1[9]) * i64::from(cospi(24)) + i64::from(step1[14]) * i64::from(cospi(8)),
    );
    step2[10] = round_shift(
        -i64::from(step1[10]) * i64::from(cospi(24)) - i64::from(step1[13]) * i64::from(cospi(8)),
    );
    step2[13] = round_shift(
        -i64::from(step1[10]) * i64::from(cospi(8)) + i64::from(step1[13]) * i64::from(cospi(24)),
    );
    step2[11] = step1[11];
    step2[12] = step1[12];
    for base in [16, 24] {
        step2[base] = step1[base] + step1[base + 3];
        step2[base + 1] = step1[base + 1] + step1[base + 2];
        step2[base + 2] = step1[base + 1] - step1[base + 2];
        step2[base + 3] = step1[base] - step1[base + 3];
        step2[base + 4] = -step1[base + 4] + step1[base + 7];
        step2[base + 5] = -step1[base + 5] + step1[base + 6];
        step2[base + 6] = step1[base + 5] + step1[base + 6];
        step2[base + 7] = step1[base + 4] + step1[base + 7];
    }

    step1[0] = step2[0] + step2[3];
    step1[1] = step2[1] + step2[2];
    step1[2] = step2[1] - step2[2];
    step1[3] = step2[0] - step2[3];
    step1[4] = step2[4];
    step1[5] = round_shift(i64::from(step2[6] - step2[5]) * i64::from(cospi(16)));
    step1[6] = round_shift(i64::from(step2[5] + step2[6]) * i64::from(cospi(16)));
    step1[7] = step2[7];
    step1[8] = step2[8] + step2[11];
    step1[9] = step2[9] + step2[10];
    step1[10] = step2[9] - step2[10];
    step1[11] = step2[8] - step2[11];
    step1[12] = -step2[12] + step2[15];
    step1[13] = -step2[13] + step2[14];
    step1[14] = step2[13] + step2[14];
    step1[15] = step2[12] + step2[15];
    step1[16] = step2[16];
    step1[17] = step2[17];
    step1[18] = round_shift(
        -i64::from(step2[18]) * i64::from(cospi(8)) + i64::from(step2[29]) * i64::from(cospi(24)),
    );
    step1[29] = round_shift(
        i64::from(step2[18]) * i64::from(cospi(24)) + i64::from(step2[29]) * i64::from(cospi(8)),
    );
    step1[19] = round_shift(
        -i64::from(step2[19]) * i64::from(cospi(8)) + i64::from(step2[28]) * i64::from(cospi(24)),
    );
    step1[28] = round_shift(
        i64::from(step2[19]) * i64::from(cospi(24)) + i64::from(step2[28]) * i64::from(cospi(8)),
    );
    step1[20] = round_shift(
        -i64::from(step2[20]) * i64::from(cospi(24)) - i64::from(step2[27]) * i64::from(cospi(8)),
    );
    step1[27] = round_shift(
        -i64::from(step2[20]) * i64::from(cospi(8)) + i64::from(step2[27]) * i64::from(cospi(24)),
    );
    step1[21] = round_shift(
        -i64::from(step2[21]) * i64::from(cospi(24)) - i64::from(step2[26]) * i64::from(cospi(8)),
    );
    step1[26] = round_shift(
        -i64::from(step2[21]) * i64::from(cospi(8)) + i64::from(step2[26]) * i64::from(cospi(24)),
    );
    for index in [22, 23, 24, 25, 30, 31] {
        step1[index] = step2[index];
    }

    for index in 0..8 {
        step2[index] = step1[index] + step1[7 - index];
        step2[8 + index] = 0;
    }
    // The first half is the ordinary 16-point stage. The assignments above
    // overlap, so spell out its difference half after all sums are available.
    step2[0] = step1[0] + step1[7];
    step2[1] = step1[1] + step1[6];
    step2[2] = step1[2] + step1[5];
    step2[3] = step1[3] + step1[4];
    step2[4] = step1[3] - step1[4];
    step2[5] = step1[2] - step1[5];
    step2[6] = step1[1] - step1[6];
    step2[7] = step1[0] - step1[7];
    step2[8] = step1[8];
    step2[9] = step1[9];
    step2[10] = round_shift(i64::from(-step1[10] + step1[13]) * i64::from(cospi(16)));
    step2[13] = round_shift(i64::from(step1[10] + step1[13]) * i64::from(cospi(16)));
    step2[11] = round_shift(i64::from(-step1[11] + step1[12]) * i64::from(cospi(16)));
    step2[12] = round_shift(i64::from(step1[11] + step1[12]) * i64::from(cospi(16)));
    step2[14] = step1[14];
    step2[15] = step1[15];
    for index in 0..8 {
        step2[16 + index] = step1[16 + index] + step1[23 - index];
        step2[24 + index] = -step1[24 + index] + step1[31 - index];
    }
    // Correct signs and order for the mirrored difference halves.
    step2[20] = step1[19] - step1[20];
    step2[21] = step1[18] - step1[21];
    step2[22] = step1[17] - step1[22];
    step2[23] = step1[16] - step1[23];
    step2[24] = -step1[24] + step1[31];
    step2[25] = -step1[25] + step1[30];
    step2[26] = -step1[26] + step1[29];
    step2[27] = -step1[27] + step1[28];
    step2[28] = step1[27] + step1[28];
    step2[29] = step1[26] + step1[29];
    step2[30] = step1[25] + step1[30];
    step2[31] = step1[24] + step1[31];

    for index in 0..8 {
        step1[index] = step2[index] + step2[15 - index];
        step1[15 - index] = step2[index] - step2[15 - index];
    }
    step1[16..20].copy_from_slice(&step2[16..20]);
    for (low, high) in [(20, 27), (21, 26), (22, 25), (23, 24)] {
        step1[low] = round_shift(i64::from(-step2[low] + step2[high]) * i64::from(cospi(16)));
        step1[high] = round_shift(i64::from(step2[low] + step2[high]) * i64::from(cospi(16)));
    }
    step1[28..32].copy_from_slice(&step2[28..32]);

    for index in 0..16 {
        output[index] = step1[index] + step1[31 - index];
        output[31 - index] = step1[index] - step1[31 - index];
    }
}

#[cfg(test)]
mod tests {
    use super::{inverse_adst, inverse_dct};

    fn reference_input(size: usize) -> [i32; 32] {
        let mut input = [0; 32];
        for (index, value) in input[..size].iter_mut().enumerate() {
            *value = ((index * 37 + size * 11) % 257) as i32 - 128;
        }
        input
    }

    #[test]
    fn zero_vectors_stay_zero() {
        let input = [0; 32];
        for size in [4, 8, 16] {
            let mut output = [1; 32];
            inverse_dct(&input, &mut output, size);
            assert!(output[..size].iter().all(|&value| value == 0));
            inverse_adst(&input, &mut output, size);
            assert!(output[..size].iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn matches_scalar_reference_vectors() {
        let references: &[(usize, &[i32], &[i32])] = &[
            (4, &[-99, -95, -9, -33], &[-58, -111, -38, -54]),
            (
                8,
                &[38, 92, -312, -51, 141, -62, -138, 64],
                &[-68, 228, -78, -232, 110, -11, -160, 42],
            ),
            (
                16,
                &[
                    189, 40, 75, 221, 591, -210, -36, 11, -125, -356, 0, -43, -129, 35, 228, 45,
                ],
                &[
                    177, -22, 57, -169, 595, 296, 1, 169, 77, -271, -97, 4, -158, 0, 225, 66,
                ],
            ),
        ];
        for &(size, expected_dct, expected_adst) in references {
            let input = reference_input(size);
            let mut output = [0; 32];
            inverse_dct(&input, &mut output, size);
            assert_eq!(&output[..size], expected_dct, "IDCT{size}");
            inverse_adst(&input, &mut output, size);
            assert_eq!(&output[..size], expected_adst, "IADST{size}");
        }

        let input = reference_input(32);
        let mut output = [0; 32];
        inverse_dct(&input, &mut output, 32);
        assert_eq!(
            output,
            [
                188, -38, 83, 47, 54, 110, 50, 276, 132, -1261, -71, -171, -53, -60, -64, 16, -110,
                258, 578, -229, -1, -93, -57, -52, -120, -20, -394, -206, 319, 7, 92, 38,
            ],
            "IDCT32"
        );
    }

    #[test]
    fn matches_scalar_reference_hashes() {
        fn hash_transform(size: usize, adst: bool) -> u64 {
            let mut state = 0x1234_5678u32;
            let mut hash = 1_469_598_103_934_665_603u64;
            let mut input = [0; 32];
            let mut output = [0; 32];
            for _ in 0..1000 {
                for value in &mut input[..size] {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = ((state >> 22) as i32) - 512;
                }
                if adst {
                    inverse_adst(&input, &mut output, size);
                } else {
                    inverse_dct(&input, &mut output, size);
                }
                for &value in &output[..size] {
                    hash ^= u64::from(value as u16);
                    hash = hash.wrapping_mul(1_099_511_628_211);
                }
            }
            hash
        }

        assert_eq!(hash_transform(4, false), 17_330_672_870_590_762_111);
        assert_eq!(hash_transform(4, true), 10_827_495_432_060_702_283);
        assert_eq!(hash_transform(8, false), 9_991_807_264_056_026_283);
        assert_eq!(hash_transform(8, true), 17_857_142_655_956_906_080);
        assert_eq!(hash_transform(16, false), 16_546_848_915_393_266_127);
        assert_eq!(hash_transform(16, true), 10_710_907_876_111_678_428);
        assert_eq!(hash_transform(32, false), 9_944_549_699_437_569_271);
    }
}
