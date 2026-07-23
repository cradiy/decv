//! Bounded storage and stable resolution of SPS/PPS versions.

use std::{array, sync::Arc};

use bit_readers::BitReader;

use crate::{H264Error, PictureParameterSet, Result, SequenceParameterSet};

#[derive(Debug, Clone)]
pub struct ActiveParameterSets {
    pub sequence: Arc<SequenceParameterSet>,
    pub picture: Arc<PictureParameterSet>,
}

#[derive(Debug, Clone)]
struct StoredPictureParameterSet {
    sequence: Arc<SequenceParameterSet>,
    picture: Arc<PictureParameterSet>,
}

#[derive(Debug)]
pub struct ParameterSetStore {
    sequence: [Option<Arc<SequenceParameterSet>>; 32],
    picture: [Option<StoredPictureParameterSet>; 256],
}

impl Default for ParameterSetStore {
    fn default() -> Self {
        Self {
            sequence: array::from_fn(|_| None),
            picture: array::from_fn(|_| None),
        }
    }
}

impl ParameterSetStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses and replaces the SPS at its bounded identifier.
    pub fn parse_sps(&mut self, rbsp: &[u8]) -> Result<Arc<SequenceParameterSet>> {
        let sps = Arc::new(SequenceParameterSet::parse(rbsp)?);
        self.sequence[sps.id as usize] = Some(Arc::clone(&sps));
        Ok(sps)
    }

    /// Parses and replaces the PPS at its bounded identifier.
    ///
    /// Each stored PPS retains the exact SPS version against which it was
    /// validated. Reusing an SPS identifier therefore cannot silently mutate
    /// the configuration of an already active picture.
    pub fn parse_pps(&mut self, rbsp: &[u8]) -> Result<Arc<PictureParameterSet>> {
        let (pps_id, sps_id) = read_parameter_set_ids(rbsp)?;
        let sequence = self
            .sequence
            .get(sps_id as usize)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(H264Error::MissingSps(sps_id))?;
        let picture = Arc::new(PictureParameterSet::parse(rbsp, &sequence)?);

        debug_assert_eq!(picture.id, pps_id);
        self.picture[pps_id as usize] = Some(StoredPictureParameterSet {
            sequence,
            picture: Arc::clone(&picture),
        });
        Ok(picture)
    }

    pub fn sequence(&self, id: u32) -> Result<Arc<SequenceParameterSet>> {
        self.sequence
            .get(id as usize)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(H264Error::MissingSps(id))
    }

    pub fn resolve(&self, picture_id: u32) -> Result<ActiveParameterSets> {
        let stored = self
            .picture
            .get(picture_id as usize)
            .and_then(Option::as_ref)
            .ok_or(H264Error::MissingPps(picture_id))?;

        Ok(ActiveParameterSets {
            sequence: Arc::clone(&stored.sequence),
            picture: Arc::clone(&stored.picture),
        })
    }

    pub fn clear(&mut self) {
        self.sequence.fill(None);
        self.picture.fill(None);
    }
}

fn read_parameter_set_ids(rbsp: &[u8]) -> Result<(u32, u32)> {
    let mut reader = BitReader::new(rbsp);
    let pps_id = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
    if pps_id > 255 {
        return Err(H264Error::InvalidSyntax("pic_parameter_set_id exceeds 255"));
    }
    let sps_id = reader.read_ue().ok_or(H264Error::UnexpectedEof)?;
    if sps_id > 31 {
        return Err(H264Error::InvalidSyntax(
            "seq_parameter_set_id in PPS exceeds 31",
        ));
    }
    Ok((pps_id, sps_id))
}

#[cfg(test)]
mod tests {
    use super::ParameterSetStore;
    use crate::H264Error;

    #[test]
    fn resolves_a_pps_with_the_sps_version_it_was_parsed_against() {
        let mut store = ParameterSetStore::new();
        store.parse_sps(&sps_rbsp(0, 4)).unwrap();
        store.parse_pps(&pps_rbsp(3, 0)).unwrap();

        let first = store.resolve(3).unwrap();
        assert_eq!(first.sequence.pic_width_in_mbs, 4);

        store.parse_sps(&sps_rbsp(0, 8)).unwrap();
        let still_first = store.resolve(3).unwrap();
        assert_eq!(still_first.sequence.pic_width_in_mbs, 4);

        store.parse_pps(&pps_rbsp(3, 0)).unwrap();
        let replaced = store.resolve(3).unwrap();
        assert_eq!(replaced.sequence.pic_width_in_mbs, 8);
    }

    #[test]
    fn reports_missing_and_out_of_range_parameter_sets() {
        let mut store = ParameterSetStore::new();

        assert!(matches!(store.resolve(9), Err(H264Error::MissingPps(9))));
        assert!(matches!(store.sequence(2), Err(H264Error::MissingSps(2))));
        assert!(matches!(
            store.parse_pps(&pps_rbsp(0, 1)),
            Err(H264Error::MissingSps(1))
        ));

        store.parse_sps(&sps_rbsp(0, 4)).unwrap();
        store.parse_pps(&pps_rbsp(0, 0)).unwrap();
        store.clear();
        assert!(matches!(store.resolve(0), Err(H264Error::MissingPps(0))));
    }

    fn sps_rbsp(id: u32, width_in_mbs: u32) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(66, 8);
        writer.write_bits(0, 8);
        writer.write_bits(30, 8);
        writer.write_ue(id);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_ue(width_in_mbs - 1);
        writer.write_ue(2);
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn pps_rbsp(id: u32, sps_id: u32) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(id);
        writer.write_ue(sps_id);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_bits(0, 2);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        current: u8,
        bits: u8,
    }

    impl BitWriter {
        fn write_flag(&mut self, value: bool) {
            self.write_bits(u64::from(value), 1);
        }

        fn write_bits(&mut self, value: u64, count: u8) {
            for shift in (0..count).rev() {
                self.current = (self.current << 1) | ((value >> shift) as u8 & 1);
                self.bits += 1;
                if self.bits == 8 {
                    self.bytes.push(self.current);
                    self.current = 0;
                    self.bits = 0;
                }
            }
        }

        fn write_ue(&mut self, value: u32) {
            let code_num = u64::from(value) + 1;
            let width = 64 - code_num.leading_zeros() as u8;
            self.write_bits(0, width - 1);
            self.write_bits(code_num, width);
        }

        fn write_se(&mut self, value: i32) {
            let code_num = if value <= 0 {
                u32::try_from(-i64::from(value) * 2).unwrap()
            } else {
                u32::try_from(i64::from(value) * 2 - 1).unwrap()
            };
            self.write_ue(code_num);
        }

        fn finish_rbsp(mut self) -> Vec<u8> {
            self.write_flag(true);
            if self.bits != 0 {
                self.current <<= 8 - self.bits;
                self.bytes.push(self.current);
            }
            self.bytes
        }
    }
}
