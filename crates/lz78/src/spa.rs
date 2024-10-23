use std::{
    fs::File,
    io::{Read, Write},
};

use crate::{
    encoder::{lz78_bits_to_encode_phrase, lz78_decode, EncodedSequence, StreamingEncoder},
    sequence::Sequence,
    tree::LZ78Tree,
    util::sample_from_pdf,
};
use anyhow::Result;
use bitvec::vec::BitVec;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

/// Streaming LZ78 encoder: you can pass in the input sequence to be compressed
/// in chunks, and the output (`encoder.get_encoded_sequence()`) is as if the
/// full concatenated sequence was passed in to an LZ78 encoder
#[derive(Clone)]
pub struct StreamingLZ78Encoder {
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

impl StreamingLZ78Encoder {
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

impl StreamingEncoder for StreamingLZ78Encoder {
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
                input,
                start_idx,
                input.len(),
                true,
                true,
            )?;

            start_idx = traversal_output.phrase_end_idx + 1;
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

/// Interface for sequential probability assignments on data
pub trait SPA {
    /// Use a block of data to update the SPA. If `include_prev_context` is
    /// true, then this block is considered to be from the same sequence as
    /// the previous. Otherwise, it is assumed to be a separate sequence (e.g.,
    /// for the LZ78 SPA, this means we start at the root of the tree).
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence;

    /// Given a fixed SPA, compute the log loss of that SPA on a test sequence
    fn compute_test_loss<T: ?Sized>(
        &mut self,
        input: &T,
        include_prev_context: bool,
    ) -> Result<f64>
    where
        T: Sequence;

    /// Computes the SPA for every symbol in the alphabet
    fn compute_spa_at_current_state(&self) -> Vec<f64>;

    /// Returns the normaliized log loss from training the SPA
    fn get_normalized_log_loss(&self) -> f64;

    /// Generates a sequence of data, using temperature and top-k sampling.
    /// For SPAs with the notion of a variable-length context, the `min_context`
    /// parameter specifies that the SPA tries to maintain a context length
    /// of at least a certain length.
    ///
    /// Using `seed_data`, you can specify that the sequence of generated data
    /// be the continuation of the specified sequence.
    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
    ) -> Result<f64>
    where
        T: Sequence;
}

/// LZ78 implementation of the sequential probability assignment
#[derive(Debug)]
pub struct LZ78SPA {
    tree: LZ78Tree,
    state: u64,
    n: u64,
    total_log_loss: f64,
    alphabet_size: u32,
}

impl LZ78SPA {
    pub fn new(alpha_size: u32, gamma: f64) -> Self {
        Self {
            tree: LZ78Tree::new_spa(alpha_size, gamma),
            state: LZ78Tree::ROOT_IDX,
            n: 0,
            total_log_loss: 0.0,
            alphabet_size: alpha_size,
        }
    }

    /// Traverses the tree  using a provided slice of the input sequence, and
    /// returns a tuple of the new state and log loss
    fn compute_test_loss_on_slice_from_state<T: Sequence + ?Sized>(
        &mut self,
        input: &T,
        state: u64,
        start_idx: u64,
        len: u64,
    ) -> Result<(u64, f64)> {
        let mut start_idx = start_idx;
        let mut log_loss = 0.0;
        let end_idx = start_idx + len;

        let mut state = state;
        while start_idx < end_idx {
            let traversal_output = self
                .tree
                .traverse_to_leaf_from(state, input, start_idx, end_idx, false, false)?;

            start_idx = traversal_output.phrase_end_idx + 1;
            log_loss += traversal_output.log_loss;
            state = if self.tree.is_leaf(traversal_output.state_idx) {
                LZ78Tree::ROOT_IDX
            } else {
                traversal_output.state_idx
            };
        }

        Ok((state, log_loss))
    }

    /// Start at the provided state, traverse the tree using the provided
    /// slice of the input sequence, stopping when we traverse past a leaf or
    /// run to the end of the input slice. returns a tuple of the new state
    /// (the root if we reached a leaf) and the log loss incurred
    fn maybe_traverse_once_to_leaf<T: Sequence>(
        &mut self,
        input: &T,
        state: u64,
        start_idx: u64,
        len: u64,
    ) -> Result<(u64, f64)> {
        let mut log_loss = 0.0;
        let end_idx = start_idx + len;

        let traversal_output = self
            .tree
            .traverse_to_leaf_from(state, input, start_idx, end_idx, false, false)?;

        log_loss += traversal_output.log_loss;
        let state = if traversal_output.reached_leaf { //self.tree.is_leaf(traversal_output.state_idx) {
            LZ78Tree::ROOT_IDX
        } else {
            traversal_output.state_idx
        };

        Ok((state, log_loss))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.n);
        bytes.put_u64_le(self.state);
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_f64_le(self.total_log_loss);
        bytes.extend(self.tree.to_bytes());
        Ok(bytes)
    }

    pub fn save_to_file(&self, path: String) -> Result<()> {
        let mut bytes = self.to_bytes()?;

        let mut file = File::create(path)?;
        file.write_all(&mut bytes)?;

        Ok(())
    }

    pub fn from_bytes(bytes: &mut Bytes) -> Result<Self> {
        let n = bytes.get_u64_le();
        let state = bytes.get_u64_le();
        let alphabet_size = bytes.get_u32_le();
        let total_log_loss = bytes.get_f64_le();
        let tree = LZ78Tree::from_bytes(bytes);

        Ok(Self {
            n,
            state,
            alphabet_size,
            total_log_loss,
            tree,
        })
    }

    pub fn from_file(path: String) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes: Vec<u8> = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut bytes: Bytes = bytes.into();
        Self::from_bytes(&mut bytes)
    }
}

impl SPA for LZ78SPA {
    /// Same as the LZ78 encoding process, but: (1) we don't actually compute
    /// the encoded bits, and (2) we compute the log loss incurred over the
    /// course of this block. By default, the LZ78Tree keeps track of the
    /// number of times each node was visited, which is sufficient to compute
    /// the SPA
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        let prev_log_loss = self.total_log_loss;
        if !include_prev_context {
            // reset the state to the root of the tree
            self.state = LZ78Tree::ROOT_IDX;
        }

        let mut start_idx = 0;
        self.n += input.len();

        while start_idx < input.len() {
            let traversal_output = self.tree.traverse_to_leaf_from(
                self.state,
                input,
                start_idx,
                input.len(),
                true,
                true,
            )?;

            start_idx = traversal_output.phrase_end_idx + 1;
            self.total_log_loss += traversal_output.log_loss;

            if traversal_output.added_leaf.is_some() {
                self.state = LZ78Tree::ROOT_IDX;
            } else {
                self.state = traversal_output.state_idx;
                break;
            }
        }

        Ok(self.total_log_loss - prev_log_loss)
    }

    /// Compute the loss of a test sequence on this SPA
    fn compute_test_loss<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        Ok(self
            .compute_test_loss_on_slice_from_state(
                input,
                if include_prev_context {
                    self.state
                } else {
                    LZ78Tree::ROOT_IDX
                },
                0,
                input.len(),
            )?
            .1)
    }

    fn compute_spa_at_current_state(&self) -> Vec<f64> {
        self.tree.compute_spa(self.state)
    }

    fn get_normalized_log_loss(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_log_loss / (self.n as f64)
        }
    }

    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut log_loss: f64 = 0.0;

        let top_k = top_k.clamp(1, self.alphabet_size);

        // by default, start at the current state of the SPA.
        let mut state = self.state;

        let mut rng = thread_rng();

        // sample from the RNG once at the beginning for efficiency
        let samples = Uniform::new(0.0, 1.0)
            .sample_iter(&mut rng)
            .take(len as usize)
            .collect_vec();

        if let Some(data) = seed_data {
            // traverse the tree using the seed data
            (state, log_loss) = self.compute_test_loss_on_slice_from_state(
                data,
                LZ78Tree::ROOT_IDX,
                0,
                data.len(),
            )?;
        }

        for sample_num in 0..len {
            // If we're at a place with no information (root or leaf), we need to
            // re-seed the SPA with some context
            if state == LZ78Tree::ROOT_IDX || self.tree.is_leaf(state) {
                // keep on trying to re-seed the SPA: first start at min_context
                // symbols from the end, and traverse the prefix tree. If we
                // reach a leaf at any point, try with min_context - 1 symbols,
                // and repeat until the traversal does not reach a leaf.
                for k in (0..=min_context.min(sample_num)).rev() {
                    state = if k == 0 {
                        // we completely failed to re-seed the SPA, so we give
                        // up and generate the next symbol from the root
                        LZ78Tree::ROOT_IDX
                    } else {
                        self.maybe_traverse_once_to_leaf(
                            output_seq,
                            LZ78Tree::ROOT_IDX,
                            sample_num - k,
                            k,
                        )?
                        .0
                    };

                    // re-seeding was successful!
                    if !self.tree.is_leaf(state) && state != LZ78Tree::ROOT_IDX {
                        break;
                    }
                }
            }

            // Compute the probability, according to the LZ78 SPA, that the
            // next symbol is x, for every x in the alphabet
            let mut spa = self.tree.compute_spa(state);

            let most_likely_next_sym = (0..self.alphabet_size)
                .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
                .unwrap();

            // if temperature is 0.0, we just compute the argmax of the SPA. If
            // temperature is 1.0, the symbols are generated directly from the
            // SPA. In either case, we do not need the following computation.
            if temperature != 0.0 && temperature != 1.0 {
                spa = spa
                    .iter()
                    .map(|x| 2.0_f64.powf(x.log2() / temperature))
                    .collect_vec();
            }

            // top-k sampling
            (0..self.alphabet_size)
                .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
                .take((self.alphabet_size - top_k) as usize)
                .map(|i| {
                    spa[i as usize] = 0.0;
                })
                .collect_vec();

            let sum: f64 = spa.iter().sum();
            spa = spa.iter().map(|x| *x / sum).collect_vec();

            let new_sym = if temperature == 0.0 {
                most_likely_next_sym
            } else {
                sample_from_pdf(&spa, samples[sample_num as usize]) as u32
            };
            output_seq.put_sym(new_sym)?;

            let new_log_loss;
            (state, new_log_loss) =
                self.compute_test_loss_on_slice_from_state(output_seq, state, sample_num, 1)?;
            log_loss += new_log_loss;
        }

        Ok(log_loss)
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, CharacterSequence, U16Sequence, U32Sequence};
    use bitvec::prelude::*;
    use itertools::Itertools;
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn sanity_check_streaming() {
        let mut all_data: Vec<u16> = Vec::new();
        let mut encoder: StreamingLZ78Encoder = StreamingLZ78Encoder::new(10);
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
    fn test_streaming_long() {
        let mut all_data: Vec<u32> = Vec::new();
        let alphabet_size = 100;
        let mut encoder: StreamingLZ78Encoder = StreamingLZ78Encoder::new(alphabet_size);
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

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut spa: LZ78SPA = LZ78SPA::new(2, 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let loss1 = spa
            .compute_test_loss(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                true,
            )
            .expect("failed to compute test loss");
        let loss2 = spa
            .compute_test_loss(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                true,
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let mut spa: LZ78SPA = LZ78SPA::new(input.alphabet_size(), 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.0,
            10,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 0, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output2 = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output2,
            100,
            10,
            1.0,
            1,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 1, topk 1, seed \"hello\": {:?}",
            generation_output2.data
        );

        assert_eq!(generation_output.data, generation_output2.data);

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            2.0,
            5,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 2, topk 5, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.5,
            10,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 0.5, topk 10, seed \"hello\": {:?}",
            generation_output.data
        );
    }
}
