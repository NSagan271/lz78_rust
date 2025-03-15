use anyhow::Result;
use itertools::Itertools;
use std::{
    fs::File,
    io::{Read, Write},
};

use bytes::{Buf, BufMut, Bytes};

pub trait ToFromBytes {
    fn to_bytes(&self) -> Result<Vec<u8>>;
    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized;

    fn save_to_file(&self, path: String) -> Result<()> {
        let mut bytes = self.to_bytes()?;

        let mut file = File::create(path)?;
        file.write_all(&mut bytes)?;

        Ok(())
    }

    fn from_file(path: String) -> Result<Self>
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

impl ToFromBytes for Vec<u64> {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.len() as u64);
        for &val in self.iter() {
            bytes.put_u64_le(val);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le();
        Ok((0..n).map(|_| bytes.get_u64_le()).collect_vec())
    }
}

impl ToFromBytes for Vec<u32> {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.len() as u64);
        for &val in self.iter() {
            bytes.put_u32_le(val);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le();
        Ok((0..n).map(|_| bytes.get_u32_le()).collect_vec())
    }
}
