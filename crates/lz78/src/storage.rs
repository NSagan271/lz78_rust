use std::{
    fs::File,
    io::{Read, Write},
};

use bytes::Bytes;

pub trait ToFromBytes {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>>;
    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn save_to_file(&self, path: String) -> anyhow::Result<()> {
        let mut bytes = self.to_bytes()?;

        let mut file = File::create(path)?;
        file.write_all(&mut bytes)?;

        Ok(())
    }

    fn from_file(path: String) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut file = File::open(path)?;
        let mut bytes: Vec<u8> = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut bytes: Bytes = bytes.into();
        Self::from_bytes(&mut bytes)
    }
}
