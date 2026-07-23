/// Whether component values use the limited or full numeric range.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorRange {
    #[default]
    Unspecified,
    Limited,
    Full,
}

/// Matrix coefficients used to convert YCbCr components.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorMatrix {
    #[default]
    Unspecified,
    Identity,
    Bt601,
    Bt709,
    Bt2020NonConstantLuminance,
}

/// Chromaticity coordinates of the source primaries.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorPrimaries {
    #[default]
    Unspecified,
    Bt601_525,
    Bt601_625,
    Bt709,
    Bt2020,
}

/// Transfer function used to encode component values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransferFunction {
    #[default]
    Unspecified,
    Linear,
    Srgb,
    Bt709,
    Bt2020TenBit,
    Bt2020TwelveBit,
}

/// Color metadata retained from codec configuration and VUI syntax.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ColorInfo {
    pub range: ColorRange,
    pub matrix: ColorMatrix,
    pub primaries: ColorPrimaries,
    pub transfer: TransferFunction,
}
