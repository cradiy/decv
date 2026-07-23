use std::io;

/// Random-access media input implemented by files, memory, HTTP range
/// requests, WebDAV range requests, caches, or encrypted storage.
pub trait MediaInput: Send + Sync {
    fn len(&self) -> io::Result<Option<u64>>;

    fn is_empty(&self) -> io::Result<Option<bool>> {
        self.len().map(|length| length.map(|length| length == 0))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize>;
}
