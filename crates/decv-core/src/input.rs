use std::{fs::File, io, sync::Arc};

/// Random-access media input implemented by files, memory, HTTP range
/// requests, WebDAV range requests, caches, or encrypted storage.
pub trait MediaInput: Send + Sync {
    fn len(&self) -> io::Result<Option<u64>>;

    fn is_empty(&self) -> io::Result<Option<bool>> {
        self.len().map(|length| length.map(|length| length == 0))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize>;
}

impl MediaInput for [u8] {
    fn len(&self) -> io::Result<Option<u64>> {
        Ok(Some(u64::try_from(self.len()).map_err(|_| {
            io::Error::other("memory input length exceeds u64")
        })?))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        let offset = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset exceeds usize"))?;
        let Some(source) = self.get(offset..) else {
            return Ok(0);
        };
        let count = source.len().min(buffer.len());
        buffer[..count].copy_from_slice(&source[..count]);
        Ok(count)
    }
}

impl MediaInput for Vec<u8> {
    fn len(&self) -> io::Result<Option<u64>> {
        MediaInput::len(self.as_slice())
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.as_slice().read_at(offset, buffer)
    }
}

impl<T> MediaInput for Arc<T>
where
    T: MediaInput + ?Sized,
{
    fn len(&self) -> io::Result<Option<u64>> {
        MediaInput::len(self.as_ref())
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.as_ref().read_at(offset, buffer)
    }
}

impl<T> MediaInput for Box<T>
where
    T: MediaInput + ?Sized,
{
    fn len(&self) -> io::Result<Option<u64>> {
        MediaInput::len(self.as_ref())
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.as_ref().read_at(offset, buffer)
    }
}

#[cfg(unix)]
impl MediaInput for File {
    fn len(&self) -> io::Result<Option<u64>> {
        self.metadata().map(|metadata| Some(metadata.len()))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buffer, offset)
    }
}

#[cfg(windows)]
impl MediaInput for File {
    fn len(&self) -> io::Result<Option<u64>> {
        self.metadata().map(|metadata| Some(metadata.len()))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buffer, offset)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::MediaInput;

    #[test]
    fn memory_inputs_read_at_offsets_without_a_cursor() {
        let input = Vec::from(*b"abcdef");
        let mut buffer = [0; 3];
        assert_eq!(input.read_at(2, &mut buffer).unwrap(), 3);
        assert_eq!(&buffer, b"cde");
        assert_eq!(input.read_at(9, &mut buffer).unwrap(), 0);
        assert_eq!(MediaInput::len(&input).unwrap(), Some(6));
    }

    #[test]
    fn owned_trait_objects_delegate_random_reads() {
        let input: Arc<dyn MediaInput> = Arc::new(Vec::from(*b"abcdef"));
        let mut buffer = [0; 2];
        assert_eq!(input.read_at(3, &mut buffer).unwrap(), 2);
        assert_eq!(&buffer, b"de");

        let input: Box<dyn MediaInput> = Box::new(Vec::from(*b"uvwxyz"));
        assert_eq!(input.read_at(1, &mut buffer).unwrap(), 2);
        assert_eq!(&buffer, b"vw");
    }
}
