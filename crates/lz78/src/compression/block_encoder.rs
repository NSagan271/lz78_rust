use anyhow::Result;
use bitvec::vec::BitVec;

use super::encoder::{lz78_bits_to_encode_phrase, lz78_decode, EncodedSequence};
use super::lzw::LZWData;
use crate::sequence::Sequence;

/// Interface for encoding blocks of a dataset in a streaming fashion; i.e.,
/// the input is passed in as several blocks.
pub trait BlockEncoder {
    fn encode_block<T>(&mut self, input: &T) -> Result<()>
    where
        T: Sequence;

    /// Returns the encoded sequence, which is the compressed version of the
    /// concatenation of all inputs to `encode_block`
    fn get_encoded_sequence(&self) -> &EncodedSequence;

    fn decode<T>(&self, output: &mut T) -> Result<()>
    where
        T: Sequence;
}

/// Streaming LZ78 encoder: you can pass in the input sequence to be compressed
/// in chunks, and the output (`encoder.get_encoded_sequence()`) is as if the
/// full concatenated sequence was passed in to an LZ78 encoder
#[derive(Clone)]
pub struct BlockLZ78Encoder {
    /// Current encoded sequence
    encoded: EncodedSequence,
    /// Current LZ78 prefix tree
    lzw: LZWData,
    /// Node of the LZ78 tree currently being traversed. This is needed for
    /// "picking up where we left off" when compressing multiple blocks
    state: u64,
    /// Number of full phrases parsed so far
    n_phrases: u64,
}

impl BlockLZ78Encoder {
    pub fn new(alpha_size: u32) -> Self {
        let bits = BitVec::new();
        let encoded = EncodedSequence::from_data(bits, 0, alpha_size);
        Self {
            encoded,
            lzw: LZWData::new(),
            state: 0,
            n_phrases: 0,
        }
    }
}

impl BlockEncoder for BlockLZ78Encoder {
    /// Encode a block of the input using LZ78 and update `self.encoded`
    fn encode_block<T>(&mut self, input: &T) -> Result<()>
    where
        T: Sequence,
    {
        let mut input_iter = input.iter().peekable();
        if self.state != 0 {
            // we were in the middle of a phrase!
            self.n_phrases -= 1;
            let last_bitwidth =
                lz78_bits_to_encode_phrase(self.n_phrases, self.encoded.alphabet_size) as u64;
            self.encoded
                .truncate(self.encoded.get_raw().len() as u64 - last_bitwidth);
        }
        self.encoded
            .set_uncompressed_len(self.encoded.uncompressed_length + input.len());

        while input_iter.peek() != None {
            let traversal_result = self.lzw.traverse_to_leaf_from(self.state, &mut input_iter);

            self.state = if traversal_result.added_leaf == None {
                traversal_result.state_idx
            } else {
                0
            };

            let leaf = traversal_result.added_leaf.unwrap_or(0);
            // value to encode, as per original LZ78 paper
            let val: u64 =
                traversal_result.state_idx * (input.alphabet_size() as u64) + (leaf as u64);

            let bitwidth = lz78_bits_to_encode_phrase(self.n_phrases, self.encoded.alphabet_size);
            self.n_phrases += 1;
            self.encoded.push(val, bitwidth);
        }

        Ok(())
    }

    fn get_encoded_sequence(&self) -> &EncodedSequence {
        &self.encoded
    }

    fn decode<T>(&self, output: &mut T) -> Result<()>
    where
        T: Sequence,
    {
        lz78_decode(output, &self.encoded)
    }
}

#[cfg(test)]
pub mod test {
    use itertools::Itertools;
    use rand::{distributions::Uniform, prelude::Distribution, thread_rng, Rng};

    use crate::sequence::{SequenceConfig, U16Sequence, U32Sequence};

    use super::*;

    #[test]
    fn sanity_check_block_encoding() {
        let mut all_data: Vec<u16> = Vec::new();
        let mut encoder: BlockLZ78Encoder = BlockLZ78Encoder::new(10);
        for _ in 0..20 {
            let new_vec = vec![
                0, 1, 2, 5, 9, 3, 4, 0, 1, 2, 5, 9, 4, 4, 4, 5, 5, 6, 7, 8, 9, 1, 2, 3,
            ];
            all_data.extend(new_vec.clone());
            let new_input = U16Sequence::from_data(new_vec, 10).expect("failed to create sequence");
            encoder.encode_block(&new_input).expect("could not encode");
        }
        let mut output = U16Sequence::new(&SequenceConfig::AlphaSize(10)).unwrap();
        encoder.decode(&mut output).expect("decoding failed");
        assert_eq!(all_data, output.data);
    }

    #[test]
    fn test_block_encoding_long() {
        let mut all_data: Vec<u32> = Vec::new();
        let alphabet_size = 100;
        let mut encoder: BlockLZ78Encoder = BlockLZ78Encoder::new(alphabet_size);
        let max_n = 10_000;

        let mut rng = thread_rng();
        for _ in 0..200 {
            let n = rng.gen_range(1..max_n);

            let new_vec = Uniform::new(0, alphabet_size)
                .sample_iter(&mut rng)
                .take(n as usize)
                .collect_vec();
            all_data.extend(new_vec.clone());
            let new_input =
                U32Sequence::from_data(new_vec, alphabet_size).expect("failed to create sequence");
            encoder.encode_block(&new_input).expect("could not encode");
        }
        let mut output = U32Sequence::new(&SequenceConfig::AlphaSize(alphabet_size)).unwrap();
        encoder.decode(&mut output).expect("decoding failed");
        assert_eq!(all_data, output.data);
    }
}
