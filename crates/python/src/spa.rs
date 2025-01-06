use crate::{Sequence, SequenceType};
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use lz78::sequence::{Sequence as RustSequence, SequenceParams};
use lz78::spa::lz_transform::{LZ78DebugState, LZ78SPA as RustLZ78SPA};
use lz78::spa::states::SPAState;
use lz78::{
    sequence::{CharacterSequence, U32Sequence, U8Sequence},
    spa::basic_spas::DirichletSPA,
    spa::generation::{generate_sequence, GenerationParams},
    spa::{SPAParams, SPA},
    storage::ToFromBytes,
};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};

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
    spa: RustLZ78SPA<DirichletSPA>,
    alphabet_size: u32,
    empty_seq_of_correct_datatype: Option<SequenceType>,
    params: SPAParams,
    state: SPAState,
}

#[pymethods]
impl LZ78SPA {
    #[new]
    #[pyo3(signature = (alphabet_size, gamma=0.5, debug=false))]
    pub fn new(alphabet_size: u32, gamma: f64, debug: bool) -> PyResult<Self> {
        let params = SPAParams::new_lz78_dirichlet(alphabet_size, gamma, debug);
        Ok(Self {
            spa: RustLZ78SPA::new(&params)?,
            state: params.get_new_state(),
            empty_seq_of_correct_datatype: None,
            alphabet_size,
            params,
        })
    }

    /// Reset the state of the LZ78 tree to the root.
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    /// Use a block of data to update the SPA. If `include_prev_context` is
    /// true, then this block is considered to be from the same sequence as
    /// the previous. Otherwise, it is assumed to be a separate sequence, and
    /// we return to the root of the LZ78 prefix tree.
    ///
    /// Returns the self-entropy log loss incurred while processing this
    /// sequence.
    #[pyo3(signature = (input))]
    pub fn train_on_block(&mut self, input: Sequence) -> PyResult<f64> {
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
            SequenceType::U8(u8_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::U8(U8Sequence::new(
                    &SequenceParams::AlphaSize(input.alphabet_size()?),
                )?));
                self.spa
                    .train_on_block(u8_sequence, &self.params, &mut self.state)?
            }
            SequenceType::Char(character_sequence) => {
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::Char(CharacterSequence::new(
                        &SequenceParams::CharMap(character_sequence.character_map.clone()),
                    )?));
                self.spa
                    .train_on_block(character_sequence, &self.params, &mut self.state)?
            }
            SequenceType::U32(u32_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::U32(U32Sequence::new(
                    &SequenceParams::AlphaSize(input.alphabet_size()?),
                )?));
                self.spa
                    .train_on_block(u32_sequence, &self.params, &mut self.state)?
            }
        })
    }

    /// Given the SPA that has been trained thus far, compute the self-entropy
    /// log loss ("perplexity") of a test sequence. `include_prev_context` has
    /// the same meaning as in `train_on_block`.
    #[pyo3(signature = (input))]
    pub fn compute_test_loss(&mut self, input: Sequence) -> PyResult<f64> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        Ok(match &input.sequence {
            SequenceType::U8(u8_sequence) => {
                self.spa
                    .test_on_block(u8_sequence, &self.params, &mut self.state)?
            }
            SequenceType::Char(character_sequence) => {
                self.spa
                    .test_on_block(character_sequence, &self.params, &mut self.state)?
            }
            SequenceType::U32(u32_sequence) => {
                self.spa
                    .test_on_block(u32_sequence, &self.params, &mut self.state)?
            }
        })
    }

    /// Computes the SPA for every symbol in the alphabet, using the LZ78
    /// context reached at the end of parsing the last training block
    pub fn compute_spa_at_current_state(&mut self) -> PyResult<Vec<f64>> {
        Ok(self.spa.spa(&self.params, &mut self.state)?)
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
    /// - seed_data: you can specify that the sequence of generated data
    ///     be the continuation of the specified sequence.
    // - temperature: a measure of how "random" the generated sequence is. A
    ///     temperature of 0 deterministically generates the most likely
    ///     symbols, and a temperature of 1 samples directly from the SPA.
    ///     Temperature values around 0.1 or 0.2 function well.
    /// - top_k: forces the generated symbols to be of the top_k most likely
    ///     symbols at each timestep.
    /// - desired_context_length: the SPA tries to maintain a context of at least a
    ///     certain length at all times. So, when we reach a leaf of the LZ78
    ///     prefix tree, we try traversing the tree with different suffixes of
    ///     the generated sequence until we get a sufficiently long context
    ///     for the next symbol.
    /// - min_spa_training_points: requires that a node of the LZ78 prefix tree
    ///     has been visited at least this number of times during training before
    ///     it can be used for generation. i.e., instead of returning to the
    ///     root upon reaching a leaf, we would return to the root once we reach
    ///     any node that has not been traversed enough times.
    ///
    /// Returns a tuple of the generated sequence and that sequence's log loss,
    /// or perplexity.
    ///
    /// Errors if the SPA has not been trained so far, or if the seed data is
    /// not over the same alphabet as the training data.
    #[pyo3(signature = (len, seed_data=None, temperature=0.5, top_k=5, desired_context_length=10, min_spa_training_points=1))]
    pub fn generate_data(
        &mut self,
        len: u64,
        seed_data: Option<Sequence>,
        temperature: f64,
        top_k: u32,
        desired_context_length: u64,
        min_spa_training_points: u64,
    ) -> PyResult<(Sequence, f64)> {
        if self.empty_seq_of_correct_datatype.is_some() && seed_data.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&seed_data.as_ref().unwrap().sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        let mut output = self.empty_seq_of_correct_datatype.clone().unwrap();

        let gen_params = GenerationParams::new(
            temperature,
            top_k,
            desired_context_length,
            min_spa_training_points,
        );

        let loss = match seed_data {
            None => match &mut output {
                SequenceType::U8(u8_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &self.params,
                    &gen_params,
                    None,
                    u8_sequence,
                )?,
                SequenceType::Char(character_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &self.params,
                    &gen_params,
                    None,
                    character_sequence,
                )?,
                SequenceType::U32(u32_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &self.params,
                    &gen_params,
                    None,
                    u32_sequence,
                )?,
            },
            Some(seed_data) => match (&mut output, &seed_data.sequence) {
                (SequenceType::U8(output_seq), SequenceType::U8(seed_seq)) => generate_sequence(
                    &mut self.spa,
                    len,
                    &self.params,
                    &gen_params,
                    Some(seed_seq),
                    output_seq,
                )?,
                (SequenceType::Char(output_seq), SequenceType::Char(seed_seq)) => {
                    generate_sequence(
                        &mut self.spa,
                        len,
                        &self.params,
                        &gen_params,
                        Some(seed_seq),
                        output_seq,
                    )?
                }
                (SequenceType::U32(output_seq), SequenceType::U32(seed_seq)) => generate_sequence(
                    &mut self.spa,
                    len,
                    &self.params,
                    &gen_params,
                    Some(seed_seq),
                    output_seq,
                )?,
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
                bytes.extend(seq.to_bytes()?);
            }
            None => bytes.put_u8(1),
        };
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.params.to_bytes()?);
        bytes.extend(self.state.to_bytes()?);
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Extracts information about the depth of leaves of the LZ78 prefix tree
    /// underlying this SPA.
    pub fn get_debug_info(&self) -> PyResult<LZ78DebugInfo> {
        if let SPAParams::LZ78(params) = &self.params {
            if !params.debug {
                return Err(PyAssertionError::new_err(
                    "Must instantiate an LZ78 SPA with debug=true to get debug info",
                ));
            }
        } else {
            unreachable!();
        }
        Ok(LZ78DebugInfo {
            debug_info: self.spa.get_debug_info().clone(),
        })
    }

    /// Resets the running list of node depths (i.e., the one returned by
    /// LZ78DebugInfo.get_depths_traversed)
    pub fn clear_debug_depths(&mut self) {
        self.spa.debug.clear_depths_traversed();
    }
}

#[pyfunction]
/// Constructs a trained SPA from its byte array representation.
pub fn spa_from_bytes<'py>(bytes: Py<PyBytes>, py: Python<'py>) -> PyResult<LZ78SPA> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    let spa: RustLZ78SPA<DirichletSPA> = RustLZ78SPA::from_bytes(&mut bytes)?;
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
    let params = SPAParams::from_bytes(&mut bytes)?;
    let state = SPAState::from_bytes(&mut bytes)?;

    Ok(LZ78SPA {
        spa,
        alphabet_size,
        empty_seq_of_correct_datatype,
        params,
        state,
    })
}

/// Debugging information for the LZ78SPA; i.e., the depths of the leaves.
/// This class cannot be instantiated, but rather is produced via
/// `LZ78SPA.get_debug_info()`
#[pyclass]
pub struct LZ78DebugInfo {
    debug_info: LZ78DebugState,
}

#[pymethods]
impl LZ78DebugInfo {
    /// Get the length of the longest branch of the LZ78 prefix tree
    fn get_max_leaf_depth(&self) -> u32 {
        self.debug_info.max_depth
    }

    /// Get the length of the shortest branch of the LZ78 prefix tree
    fn get_min_leaf_depth(&self) -> u32 {
        *self.debug_info.leaf_depths.values().min().unwrap_or(&0)
    }

    /// Get the (unweighted) average depth of all leaves of the LZ78 prefix tree
    fn get_mean_leaf_depth(&self) -> f64 {
        self.debug_info
            .leaf_depths
            .values()
            .map(|&x| x as f64)
            .sum::<f64>()
            / self.debug_info.leaf_depths.len() as f64
    }

    /// Returns the depth of all leaves of the LZ78 tree as a list (in no
    /// particular order)
    fn get_leaf_depths(&self) -> Vec<u32> {
        self.debug_info
            .leaf_depths
            .values()
            .map(|&x| x)
            .collect_vec()
    }

    /// Returns the longest LZ78 phrase encoded by the prefix tree,
    /// as a list of integer symbols
    fn get_longest_branch(&self) -> Vec<u32> {
        self.debug_info.get_longest_branch()
    }

    /// Returns the depth of the currently-traversed node in the LZ tree,
    /// for each timepoint. Does not distinguish between training, inference,
    /// and generation; it is recommended to use `spa.clear_debug_depths()`
    /// between training and inference or generation.
    fn get_depths_traversed(&self) -> Vec<u32> {
        self.debug_info.depths_traversed.clone()
    }
}
