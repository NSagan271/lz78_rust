use bytes::Bytes;

pub trait ToFromBytes {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>>;
    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized;
}
