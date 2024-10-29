use bytes::{Buf, BufMut, Bytes};
use lz78::{
    sequence::{CharacterSequence, U32Sequence, U8Sequence},
    spa::SPA,
};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};

use crate::{Sequence, SequenceType};

/// Constructs a sequential probability assignment on input data via LZ78
/// incremental parsing. This is the implementation of the family of SPAs
/// described in "A Family of LZ78-based Universal Sequential Probability
/// Assignments" (Sagan and Weissman, 2024), under a Dirichlet(gamma) prior.
///
/// Under this prior, the sequential probability assignment is an additive
/// perturbation of the emprical distribution, conditioned on the LZ78 prefix
/// of each symbol (i.e., the probability model is proportional to the
/// number of times each node of the LZ78 tree has been visited, plus gamma).
///
/// This SPA has the following capabilities:
/// - training on one or more sequences,
/// - log loss ("perplexity") computation for test sequences,
/// - SPA computation (using the LZ78 context reached at the end of parsing
///     the last training block),
/// - sequence generation.
///
/// Note that the LZ78SPA does not perform compression; you would have to use
/// a separate BlockLZ78Encoder object to perform block-wise compression.
#[pyclass]
pub struct LZ78SPA {
    spa: lz78::spa::LZ78SPA,
    alphabet_size: u32,
    empty_seq_of_correct_datatype: Option<SequenceType>,
}

#[pymethods]
impl LZ78SPA {
    #[new]
    #[pyo3(signature = (alphabet_size, gamma=0.5))]
    pub fn new(alphabet_size: u32, gamma: f64) -> PyResult<Self> {
        Ok(Self {
            spa: lz78::spa::LZ78SPA::new(alphabet_size, gamma),
            empty_seq_of_correct_datatype: None,
            alphabet_size,
        })
    }

    /// Use a block of data to update the SPA. If `include_prev_context` is
    /// true, then this block is considered to be from the same sequence as
    /// the previous. Otherwise, it is assumed to be a separate sequence, and
    /// we return to the root of the LZ78 prefix tree.
    ///
    /// Returns the self-entropy log loss incurred while processing this
    /// sequence.
    #[pyo3(signature = (input, include_prev_context=false))]
    pub fn train_on_block(&mut self, input: Sequence, include_prev_context: bool) -> PyResult<f64> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else if self.alphabet_size != input.alphabet_size()? {
            return Err(PyAssertionError::new_err(format!(
                "Expected alphabet size of {}, got {}",
                self.alphabet_size,
                input.alphabet_size()?
            )));
        }
        Ok(match &input.sequence {
            crate::SequenceType::U8(u8_sequence) => {
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U8(U8Sequence::new(input.alphabet_size()?)));
                self.spa.train_on_block(u8_sequence, include_prev_context)?
            }
            crate::SequenceType::Char(character_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::Char(
                    CharacterSequence::new(character_sequence.character_map.clone()),
                ));
                self.spa
                    .train_on_block(character_sequence, include_prev_context)?
            }
            crate::SequenceType::U32(u32_sequence) => {
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U32(U32Sequence::new(input.alphabet_size()?)));
                self.spa
                    .train_on_block(u32_sequence, include_prev_context)?
            }
        })
    }

    /// Given the SPA that has been trained thus far, compute the self-entropy
    /// log loss ("perplexity") of a test sequence. `include_prev_context` has
    /// the same meaning as in `train_on_block`.
    #[pyo3(signature = (input, include_prev_context=false))]
    pub fn compute_test_loss(
        &mut self,
        input: Sequence,
        include_prev_context: bool,
    ) -> PyResult<f64> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        Ok(match &input.sequence {
            crate::SequenceType::U8(u8_sequence) => self
                .spa
                .compute_test_loss(u8_sequence, include_prev_context)?,
            crate::SequenceType::Char(character_sequence) => self
                .spa
                .compute_test_loss(character_sequence, include_prev_context)?,
            crate::SequenceType::U32(u32_sequence) => self
                .spa
                .compute_test_loss(u32_sequence, include_prev_context)?,
        })
    }

    /// Computes the SPA for every symbol in the alphabet, using the LZ78
    /// context reached at the end of parsing the last training block
    pub fn compute_spa_at_current_state(&self) -> Vec<f64> {
        self.spa.compute_spa_at_current_state()
    }

    /// Returns the normaliized self-entropy log loss incurred from training
    /// the SPA thus far.
    pub fn get_normalized_log_loss(&self) -> f64 {
        self.spa.get_normalized_log_loss()
    }

    /// Generates a sequence of data, using temperature and top-k sampling (see
    /// the "Experiments" section of [Sagan and Weissman 2024] for more details).
    ///
    /// Inputs:
    /// - len: number of symbols to generate
    /// - min_context: the SPA tries to maintain a context of at least a
    ///     certain length at all times. So, when we reach a leaf of the LZ78
    ///     prefix tree, we try traversing the tree with different suffixes of
    ///     the generated sequence until we get a sufficiently long context
    ///     for the next symbol.
    /// - temperature: a measure of how "random" the generated sequence is. A
    ///     temperature of 0 deterministically generates the most likely
    ///     symbols, and a temperature of 1 samples directly from the SPA.
    ///     Temperature values around 0.1 or 0.2 function well.
    /// - top_k: forces the generated symbols to be of the top_k most likely
    ///     symbols at each timestep.
    /// - seed_data: you can specify that the sequence of generated data
    /// be the continuation of the specified sequence.
    ///
    /// Returns a tuple of the generated sequence and that sequence's log loss,
    /// or perplexity.
    ///
    /// Errors if the SPA has not been trained so far, or if the seed data is
    /// not over the same alphabet as the training data.
    #[pyo3(signature = (len, min_context=0, temperature=0.1, top_k=10, seed_data=None))]
    pub fn generate_data(
        &mut self,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<Sequence>,
    ) -> PyResult<(Sequence, f64)> {
        if self.empty_seq_of_correct_datatype.is_none() {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }
        if self.empty_seq_of_correct_datatype.is_some() && seed_data.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&seed_data.as_ref().unwrap().sequence)?;
        }

        let mut output = self.empty_seq_of_correct_datatype.clone().unwrap();

        let loss = match seed_data {
            None => match &mut output {
                SequenceType::U8(u8_sequence) => self.spa.generate_data(
                    u8_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
                SequenceType::Char(character_sequence) => self.spa.generate_data(
                    character_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
                SequenceType::U32(u32_sequence) => self.spa.generate_data(
                    u32_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
            },
            Some(seed_data) => match (&mut output, &seed_data.sequence) {
                (SequenceType::U8(output_seq), SequenceType::U8(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                (SequenceType::Char(output_seq), SequenceType::Char(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                (SequenceType::U32(output_seq), SequenceType::U32(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                _ => return Err(PyAssertionError::new_err("Unexpected seed data type")),
            },
        };

        Ok((Sequence { sequence: output }, loss))
    }

    /// Returns a byte array representing the trained SPA, e.g., to save the
    /// SPA to a file.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes = self.spa.to_bytes()?;
        match &self.empty_seq_of_correct_datatype {
            Some(seq) => {
                bytes.put_u8(0);
                bytes.extend(seq.to_bytes());
            }
            None => bytes.put_u8(1),
        };
        bytes.put_u32_le(self.alphabet_size);
        Ok(PyBytes::new_bound(py, &bytes))
    }
}

#[pyfunction]
/// Constructs a trained SPA from its byte array representation.
pub fn spa_from_bytes<'py>(bytes: Py<PyBytes>, py: Python<'py>) -> PyResult<LZ78SPA> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    let spa = lz78::spa::LZ78SPA::from_bytes(&mut bytes)?;
    let empty_seq_of_correct_datatype = match bytes.get_u8() {
        0 => Some(SequenceType::from_bytes(&mut bytes)?),
        1 => None,
        _ => {
            return Err(PyAssertionError::new_err(
                "Error reading encoded sequence from bytes",
            ))
        }
    };
    let alphabet_size = bytes.get_u32_le();

    Ok(LZ78SPA {
        spa,
        alphabet_size,
        empty_seq_of_correct_datatype,
    })
}
