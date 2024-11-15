use anyhow::Result;
use bitvec::field::BitField;
use bitvec::vec::BitVec;
use bytes::{Buf, BufMut, Bytes};

use crate::lzw::LZWData;
use crate::sequence::Sequence;

/// Stores an encoded bitstream, as well as the alphabet size and length of the
/// original sequence
#[derive(Debug, Clone)]
pub struct EncodedSequence {
    data: BitVec<u64>,
    pub uncompressed_length: u64,
    pub alphabet_size: u32,
}

impl EncodedSequence {
    pub fn from_data(data: BitVec<u64>, uncompressed_length: u64, alphabet_size: u32) -> Self {
        Self {
            data,
            uncompressed_length,
            alphabet_size,
        }
    }

    /// This is only true if the uncompressed filesize is equal to the number
    /// of symbols times log2(alphabet size). In other cases (e.g., unicode
    /// text, where symbols can be from 1 to 4 bytes), you should divide the
    /// compressed length by the known uncompressed size.
    pub fn compression_ratio(&self) -> f32 {
        if self.uncompressed_length == 0 {
            0.0
        } else {
            (self.data.len() as f32)
                / (self.uncompressed_length as f32 * (self.alphabet_size as f32).log2())
        }
    }

    /// Length of the bitstream, rounded up to the nearest byte
    pub fn compressed_len_bytes(&self) -> u64 {
        (self.data.len() as u64 + 7) / 8
    }

    // In-place, truncates the underlying compressed bitstream
    pub fn truncate(&mut self, num_bits: u64) {
        self.data.drain(num_bits as usize..);
    }

    /// Pushes some bits to the bitstream
    pub fn push(&mut self, val: u64, bitwidth: u16) {
        let old_len = self.data.len();
        self.data.resize(self.data.len() + bitwidth as usize, false);
        self.data[old_len..].store_le(val);
    }

    pub fn set_uncompressed_len(&mut self, len: u64) {
        self.uncompressed_length = len;
    }

    pub fn extend_capacity(&mut self, n: u64) {
        self.data.reserve(n as usize);
    }

    /// Returns a reference to the underlying data array
    pub fn get_raw(&self) -> &BitVec<u64> {
        &self.data
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_u64_le(self.uncompressed_length);
        bytes.put_u64_le(self.data.len() as u64);
        for chunk in self.data.chunks(8) {
            bytes.put_u8(chunk.load_le::<u8>())
        }

        bytes
    }

    pub fn from_bytes(bytes: &mut Bytes) -> Self {
        let alphabet_size = bytes.get_u32_le();
        let uncompressed_length = bytes.get_u64_le();
        let data_len = bytes.get_u64_le();

        let mut data = BitVec::with_capacity(data_len as usize);
        data.resize(data_len as usize, false);
        let mut i = 0;
        for _ in 0..(data_len + 7) / 8 {
            data[i..(data_len as usize).min(i + 8)].store_le::<u8>(bytes.get_u8());
            i += 8;
        }

        Self {
            alphabet_size,
            uncompressed_length,
            data,
        }
    }
}

/// Generic interface for encoding and decoding. Anything that implements
/// this trait can be used to encode or decode sequences.
pub trait Encoder {
    fn encode<T>(&self, input: &T) -> Result<EncodedSequence>
    where
        T: Sequence;

    fn decode<T>(&self, output: &mut T, input: &EncodedSequence) -> Result<()>
    where
        T: Sequence;
}

/// LZ78 encoder implementation
pub struct LZ8Encoder {}

impl Encoder for LZ8Encoder {
    fn encode<T>(&self, input: &T) -> Result<EncodedSequence>
    where
        T: Sequence,
    {
        lz78_encode(input)
    }

    fn decode<T>(&self, output: &mut T, input: &EncodedSequence) -> Result<()>
    where
        T: Sequence,
    {
        lz78_decode(output, input)
    }
}

impl LZ8Encoder {
    pub fn new() -> Self {
        Self {}
    }
}

/// returns the number of bits required to encode the specified phrase
pub fn lz78_bits_to_encode_phrase(phrase_idx: u64, alpha_size: u32) -> u16 {
    (((phrase_idx + 1) as f64).log2() + (alpha_size as f64).log2()).ceil() as u16
}

/// Compresses a sequence using LZ78
pub fn lz78_encode<T>(input: &T) -> Result<EncodedSequence>
where
    T: Sequence,
{
    // LZ78 prefix tree that is being built during the encoding process
    let mut lzw: LZWData = LZWData::new();

    let mut bits: BitVec<u64> = BitVec::new();
    let mut prev_bit_idx = 0;

    let mut input_iter = input.iter().peekable();
    let mut phrase_num = 0;
    while input_iter.peek() != None {
        let traversal_result = lzw.traverse_root_to_leaf(&mut input_iter);
        let bitwidth = lz78_bits_to_encode_phrase(phrase_num as u64, input.alphabet_size());
        phrase_num += 1;

        let leaf = traversal_result.added_leaf.unwrap_or(0);

        // value to encode, as per original LZ78 paper
        let val: u64 = traversal_result.state_idx * (input.alphabet_size() as u64) + (leaf as u64);
        bits.resize(prev_bit_idx + bitwidth as usize, false);
        bits[prev_bit_idx..prev_bit_idx + bitwidth as usize].store_le(val);
        prev_bit_idx += bitwidth as usize;
    }

    Ok(EncodedSequence::from_data(
        bits,
        input.len(),
        input.alphabet_size(),
    ))
}

/// Decodes a sequence using LZ78
pub fn lz78_decode<T>(output: &mut T, input: &EncodedSequence) -> Result<()>
where
    T: Sequence,
{
    // the indices at which phrases start in the uncompressed sequence.
    // The first phrase, by convention, is the empty phrase.
    let mut phrase_starts: Vec<u64> = vec![0];
    // how long each phrase is (can be computed from phrase_starts, but this
    // makes decoding easier).
    let mut phrase_lengths: Vec<u64> = vec![0];

    // number of bits that have been decoded so far
    let mut bits_decoded: usize = 0;

    let alphabet_size = input.alphabet_size;

    while bits_decoded < input.data.len() {
        // number of bits that were used to store the current phrase
        let bitwidth: i32 =
            lz78_bits_to_encode_phrase(phrase_starts.len() as u64 - 1, alphabet_size) as i32;

        let decoded_val =
            input.data[bits_decoded..bits_decoded + bitwidth as usize].load_le::<u64>();
        bits_decoded += bitwidth as usize;

        // find the index of the previous phrase that forms the prefix of the
        // current phrase
        let ref_idx = decoded_val / (alphabet_size as u64);
        // the final symbol of the phrase
        let new_sym = (decoded_val % (alphabet_size as u64)) as u32;

        let phrase_start = output.len();
        let phrase_len = phrase_lengths[ref_idx as usize] + 1;
        let copy_start = phrase_starts[ref_idx as usize];

        for j in 0..phrase_len - 1 {
            output.put_sym(output.try_get(copy_start + j)?)?;
            if output.len() >= input.uncompressed_length {
                return Ok(());
            }
        }

        output.put_sym(new_sym)?;
        phrase_lengths.push(phrase_len);
        phrase_starts.push(phrase_start);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, CharacterMap, CharacterSequence, U8Sequence};
    use bitvec::prelude::*;
    use itertools::Itertools;
    use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

    use super::*;

    #[test]
    fn test_encode_decode_binary() {
        let input = BinarySequence::from_data(bitvec![
            0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        ]);
        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = BinarySequence::new();
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }

    #[test]
    fn test_encode_decode_string() {
        let charmap = CharacterMap::from_data(&"abcdefghijklmnopqrstuvwxyz ".to_string());
        let input = CharacterSequence::from_data(
            "hello hello hello hello hello world hello world hello world wxyz".to_string(),
            charmap.clone(),
        )
        .expect("input sequence invalid");
        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = CharacterSequence::new(charmap);
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }

    #[test]
    fn test_encode_long_sequence() {
        let alphabet_size = 100;
        let n = 1_000_000;

        let mut rng = thread_rng();
        let input = U8Sequence::from_data(
            Uniform::new(0, alphabet_size)
                .sample_iter(&mut rng)
                .take(n as usize)
                .collect_vec(),
            alphabet_size as u32,
        )
        .expect("creating sequence failed");

        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = U8Sequence::new(alphabet_size as u32);
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }

    #[test]
    fn test_encoded_sequence_to_from_bytes() {
        let alphabet_size = 100;
        let n = 500;

        let mut rng = thread_rng();
        let input = U8Sequence::from_data(
            Uniform::new(0, alphabet_size)
                .sample_iter(&mut rng)
                .take(n as usize)
                .collect_vec(),
            alphabet_size as u32,
        )
        .expect("creating sequence failed");

        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut bytes = encoded.to_bytes();
        // add some random bytes to the end because this shouldn't affect
        // decoding
        bytes.extend(vec![
            0, 1, 1, 0, 1, 1, 1, 43, 56, 128, 32, 255, 123, 6, 7, 8,
        ]);
        let mut bytes: Bytes = bytes.into();
        let encoded_hat = EncodedSequence::from_bytes(&mut bytes);
        assert_eq!(encoded.alphabet_size, encoded_hat.alphabet_size);
        assert_eq!(encoded.uncompressed_length, encoded_hat.uncompressed_length);
        assert_eq!(encoded.data, encoded_hat.data);
    }
}
