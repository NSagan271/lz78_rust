use anyhow::Result;
use bitvec::vec::BitVec;

use crate::{
    encoder::{lz78_bits_to_encode_phrase, lz78_decode, EncodedSequence},
    sequence::{Sequence, SequenceSlice},
    tree::LZ78Tree,
};

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
    tree: LZ78Tree,
    /// Node of the LZ78 tree currently being traversed. This is needed for
    /// "picking up where we left off" when compressing multiple blocks
    state: u64,
    /// Number of symbols compressed thus far
    n: u64,
    /// Number of full phrases parsed so far
    n_phrases: u64,
    /// How many of the output bits in `encoded` correspond to finished phrases,
    /// i.e., ones where a leaf was added to the LZ78 tree
    n_output_bits_finished_phrases: u64,
}

impl BlockLZ78Encoder {
    pub fn new(alpha_size: u32) -> Self {
        let bits = BitVec::new();
        let encoded = EncodedSequence::from_data(bits, 0, alpha_size);
        Self {
            encoded,
            tree: LZ78Tree::new(alpha_size),
            state: LZ78Tree::ROOT_IDX,
            n: 0,
            n_phrases: 0,
            n_output_bits_finished_phrases: 0,
        }
    }
}

impl BlockEncoder for BlockLZ78Encoder {
    /// Encode a block of the input using LZ78 and update `self.encoded`
    fn encode_block<T>(&mut self, input: &T) -> Result<()>
    where
        T: Sequence,
    {
        let mut start_idx = 0;

        let mut ref_idxs: Vec<u64> = Vec::new();
        let mut output_leaves: Vec<u32> = Vec::new();

        // whether we leave off in the middle of parsing a phrase
        let mut parsing_in_progress = false;

        self.n += input.len();

        while start_idx < input.len() {
            let traversal_output = self.tree.traverse_to_leaf_from(
                self.state,
                SequenceSlice::new(input, start_idx, input.len() - start_idx),
                true,
            )?;

            start_idx += traversal_output.phrase_prefix_len + 1;
            ref_idxs.push(traversal_output.state_idx);
            output_leaves.push(traversal_output.added_leaf.unwrap_or(0));

            if traversal_output.added_leaf.is_some() {
                self.state = LZ78Tree::ROOT_IDX;
            } else {
                self.state = traversal_output.state_idx;
                parsing_in_progress = true;
                break;
            }
        }

        let mut n_output_bits = 0;

        // the number of encoded bits, except perhaps for the final phrase (if
        // the final phrase is not a full phrase)
        let mut n_output_bits_finished_phrases = 0;
        for i in 0..(output_leaves.len() as u64) {
            n_output_bits_finished_phrases = n_output_bits;
            n_output_bits +=
                lz78_bits_to_encode_phrase(i + self.n_phrases, input.alphabet_size()) as u64;
        }

        let mut n_full_phrases = self.n_phrases + (output_leaves.len() - 1) as u64;
        if !parsing_in_progress {
            // the parsing ends right at the end of a phrase
            n_full_phrases += 1;
            n_output_bits_finished_phrases = n_output_bits;
        }

        // complete encoding
        self.encoded.set_uncompressed_len(self.n);
        // if there was an unfinished phrase at th end of `self.encoded`,
        // delete the bits corresponding to it, because it's included in the
        // output of this block
        self.encoded.truncate(self.n_output_bits_finished_phrases);
        // allocate memory once for performance reasons
        self.encoded.extend_capacity((n_output_bits + 7) / 8);

        // Encoding, as per `lz78_encode`
        for (i, (leaf, ref_idx)) in output_leaves.into_iter().zip(ref_idxs).enumerate() {
            let bitwidth =
                lz78_bits_to_encode_phrase(i as u64 + self.n_phrases, input.alphabet_size());
            let val = if i == 0 && self.n_phrases == 0 {
                leaf as u64
            } else {
                ref_idx * (input.alphabet_size() as u64) + (leaf as u64)
            };

            self.encoded.push(val, bitwidth);
        }

        self.n_output_bits_finished_phrases += n_output_bits_finished_phrases;
        self.n_phrases = n_full_phrases;

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

    use crate::sequence::{U16Sequence, U32Sequence};

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
        let mut output = U16Sequence::new(10);
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
        let mut output = U32Sequence::new(alphabet_size);
        encoder.decode(&mut output).expect("decoding failed");
        assert_eq!(all_data, output.data);
    }
}
