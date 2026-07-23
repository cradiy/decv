use std::sync::Arc;

use crate::MediaTime;

/// A compressed video packet with complete decode and presentation timing.
#[derive(Debug, Clone)]
pub struct EncodedVideoPacket {
    pub data: Arc<[u8]>,
    pub pts: Option<MediaTime>,
    pub dts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub keyframe: bool,
    pub discontinuity: bool,
}

impl EncodedVideoPacket {
    pub fn new(data: impl Into<Arc<[u8]>>) -> Self {
        Self {
            data: data.into(),
            pts: None,
            dts: None,
            duration: None,
            keyframe: false,
            discontinuity: false,
        }
    }
}
