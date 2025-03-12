use std::fs::File;
use std::io::Read;

use crate::{Sequence, SequenceType};
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use lz78::sequence::{Sequence as RustSequence, SequenceParams};
use lz78::spa::dirichlet::DirichletSPATree;
use lz78::spa::lz_transform::LZ78SPA as RustLZ78SPA;
use lz78::spa::params::{
    AdaptiveGamma, BackshiftParsing, DirichletParamsBuilder, Ensemble, LZ78ParamsBuilder, SPAParams,
};
use lz78::spa::states::SPAState;
use lz78::spa::util::LbAndTemp;
use lz78::{
    sequence::{CharacterSequence, U32Sequence, U8Sequence},
    spa::generation::generate_sequence,
    spa::SPA,
    storage::ToFromBytes,
};
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

fn get_param_dict<'py>(
    py: Python<'py>,
    params: &SPAParams,
    generation: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params = params.try_get_lz78()?;
    let inner_params = params.inner_params.try_get_dirichlet()?;

    let (lb, temp, lb_or_temp) = match inner_params.lb_and_temp {
        LbAndTemp::TempFirst { lb, temp } => (lb, temp, "temp_first"),
        LbAndTemp::LbFirst { lb, temp } => (lb, temp, "lb_first"),
        LbAndTemp::Skip => (0., 1., "Disabled"),
    };

    let (n_ensemble, ensemble) = match params.ensemble {
        Ensemble::Average(n) => (n, "Average"),
        Ensemble::Entropy(n) => (n, "Entropy"),
        Ensemble::Depth(n) => (n, "Depth"),
        Ensemble::None => (1, "Disabled"),
    };

    let adaptive_gamma = match params.adaptive_gamma {
        AdaptiveGamma::Inverse => "Inverse",
        AdaptiveGamma::Count => "Count",
        AdaptiveGamma::None => "Disabled",
    };

    let (bs_parsing, ctx_len, min_count, break_at_phrase) = match params.backshift_parsing {
        BackshiftParsing::Enabled {
            desired_context_length,
            min_spa_training_points,
            break_at_phrase,
        } => (
            true,
            desired_context_length,
            min_spa_training_points,
            break_at_phrase,
        ),
        BackshiftParsing::Disabled => (false, 0, 0, false),
    };

    let mut key_vals: Vec<(&str, PyObject)> = vec![
        ("gamma", inner_params.gamma.to_object(py)),
        ("adaptive_gamma", adaptive_gamma.to_object(py)),
        ("n_ensemble", n_ensemble.to_object(py)),
        ("ensemble_type", ensemble.to_object(py)),
        ("parallel_ensemble", params.par_ensemble.to_object(py)),
        ("backshift_parsing", bs_parsing.to_object(py)),
        ("backshift_ctx_len", ctx_len.to_object(py)),
        ("backshift_min_count", min_count.to_object(py)),
        ("backshift_break_at_phrase", break_at_phrase.to_object(py)),
    ];
    if !generation {
        key_vals.extend(vec![
            ("lb", lb.to_object(py)),
            ("temp", temp.to_object(py)),
            ("lb_or_temp_first", lb_or_temp.to_object(py)),
        ]);
    }
    let dict = key_vals.into_py_dict_bound(py);

    Ok(dict)
}

fn set_all_parameters(
    params: &mut SPAParams,
    gamma: Option<f64>,
    lb: Option<f64>,
    temp: Option<f64>,
    lb_or_temp_first: Option<&str>,
    adaptive_gamma: Option<&str>,
    ensemble_type: Option<&str>,
    ensemble_n: Option<u32>,
    parallel_ensemble: Option<bool>,
    backshift_parsing: Option<bool>,
    backshift_ctx_len: Option<u64>,
    backshift_min_count: Option<u64>,
    backshift_break_at_phrase: Option<bool>,
) -> PyResult<()> {
    let params = params.try_get_lz78_mut()?;
    let mut inner_params = params.inner_params.try_get_dirichlet_mut()?.clone();
    if let Some(gamma) = gamma {
        inner_params.gamma = gamma;
    }

    let (curr_lb, curr_temp) = inner_params.lb_and_temp.get_vals();
    let lb = lb.unwrap_or(curr_lb);
    let temp = temp.unwrap_or(curr_temp);

    if let Some(lb_or_temp) = lb_or_temp_first {
        inner_params.lb_and_temp = match lb_or_temp.to_lowercase().as_str() {
            "temp_first" => LbAndTemp::TempFirst { lb, temp },
            "lb_first" => LbAndTemp::LbFirst { lb, temp },
            "disabled" => LbAndTemp::Skip,
            _ => {
                return Err(PyAssertionError::new_err(
                    "Unexpected value of lb_or_temp_first",
                ))
            }
        };
    } else {
        inner_params.lb_and_temp.set_vals(lb, temp)?;
    }

    if let Some(ad_gamma) = adaptive_gamma {
        params.adaptive_gamma = match ad_gamma.to_lowercase().as_str() {
            "inverse" => AdaptiveGamma::Inverse,
            "count" => AdaptiveGamma::Count,
            "disabled" => AdaptiveGamma::None,
            _ => {
                return Err(PyAssertionError::new_err(
                    "Unexpected value of adaptive_gamma",
                ));
            }
        }
    }

    let curr_ens_n = params.ensemble.get_num_states() as u32;
    let ensemble_n = ensemble_n.unwrap_or(curr_ens_n);
    if let Some(ens) = ensemble_type {
        params.ensemble = match ens.to_lowercase().as_str() {
            "average" => Ensemble::Average(ensemble_n),
            "entropy" => Ensemble::Entropy(ensemble_n),
            "depth" => Ensemble::Depth(ensemble_n),
            "disabled" => Ensemble::None,
            _ => {
                return Err(PyAssertionError::new_err("Unexpected value of esemble"));
            }
        }
    } else {
        params.ensemble.set_num_states(ensemble_n)?;
    }

    if let Some(par_ens) = parallel_ensemble {
        params.par_ensemble = par_ens;
    }

    let backshift_parsing =
        backshift_parsing.unwrap_or(params.backshift_parsing != BackshiftParsing::Disabled);
    if !backshift_parsing {
        if backshift_ctx_len.is_some() || backshift_min_count.is_some() {
            return Err(PyAssertionError::new_err("Tried to set \"backshift_ctx_len\" or \"backshift_min_count\" while \"backshift_parsing\" = False. Please set \"backshift_parsing\" to True."));
        }
        params.backshift_parsing = BackshiftParsing::Disabled;
    } else {
        let (old_ctx_len, old_min, old_break_at_phrase) = params.backshift_parsing.get_params();
        let backshift_ctx_len = backshift_ctx_len.unwrap_or(old_ctx_len);
        let backshift_min_count = backshift_min_count.unwrap_or(old_min);
        let backshift_break_at_phrase = backshift_break_at_phrase.unwrap_or(old_break_at_phrase);
        params.backshift_parsing = BackshiftParsing::Enabled {
            desired_context_length: backshift_ctx_len,
            min_spa_training_points: backshift_min_count,
            break_at_phrase: backshift_break_at_phrase,
        };
    }
    params.inner_params = Box::new(SPAParams::Dirichlet(inner_params));

    Ok(())
}

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
    spa: RustLZ78SPA<DirichletSPATree>,
    alphabet_size: u32,
    empty_seq_of_correct_datatype: Option<SequenceType>,
    params: SPAParams,
    gen_params: SPAParams,
    state: SPAState,
}

#[pymethods]
impl LZ78SPA {
    #[new]
    #[pyo3(signature = (alphabet_size, gamma=0.5, debug=false, compute_training_loss=true))]
    pub fn new(
        alphabet_size: u32,
        gamma: f64,
        debug: bool,
        compute_training_loss: bool,
    ) -> PyResult<Self> {
        let params = LZ78ParamsBuilder::new(
            DirichletParamsBuilder::new(alphabet_size)
                .gamma(gamma)
                .lb_and_temp(1e-4, 1.0, true)
                .compute_training_log_loss(compute_training_loss)
                .build_enum(),
        )
        .backshift(5, 1, false)
        .debug(debug)
        .build_enum();

        let gen_params = LZ78ParamsBuilder::new(
            DirichletParamsBuilder::new(alphabet_size)
                .gamma(gamma)
                .build_enum(),
        )
        .backshift(5, 1, false)
        .debug(debug)
        .build_enum();

        Ok(Self {
            spa: RustLZ78SPA::new(&params)?,
            state: params.get_new_state(),
            empty_seq_of_correct_datatype: None,
            alphabet_size,
            params,
            gen_params,
        })
    }

    /// Returns a dictionary of all of the LZ hyperparameters being used for
    /// inference. See the docstring of self.set_inference_params for
    /// descriptions of each parameter.
    pub fn get_inference_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        get_param_dict(py, &self.params, false)
    }

    /// Returns a dictionary of all of the LZ hyperparameters being used for
    /// generation. See the docstring of self.set_generation_params for
    /// descriptions of each parameter.
    pub fn get_generation_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        get_param_dict(py, &self.gen_params, true)
    }

    /// Sets the hyperparameters used for inference and SPA computation. Pass
    /// in a value to change it; otherwise, values will remain at their current
    /// values by default (see self.get_inference_params for the current
    /// parameter values).
    ///
    /// - gamma: the Dirichlet smoothing hyperparameters for computing the SPA
    ///
    /// - lb: a lower bound on the SPA value for any symbol; applied only if
    ///     lb_or_temp_first is not "disabled".
    ///
    /// - temp: temperature, applied by modifying the SPA to be
    ///     softmax(2^(spa / temp)); applied only if lb_or_temp_first is not
    ///     "disabled".
    ///
    /// - lb_or_temp_first: either "temp_first", "lb_first", or "disabled".
    ///
    /// - adaptive_gamma: whether to scale gamma to be smaller for deeper
    ///     nodes, or for nodes that have seen fewer symbols.
    ///     
    ///     Possible Values: either "inverse" (for depth-based adaptive gamma),
    ///      "count", or "disabled".
    ///
    /// - ensemble_type: type of ensemble inference to use.
    ///
    ///     Possible Values: "average" to average the ensemble SPAs, "entropy"
    ///     to weight the average based on the entropy of each SPA, "depth" to
    ///     weight the average based on the node depths, or "disabled".
    ///
    /// - ensemble_n: number of nodes in the ensemble; only valid if
    ///     "ensemble_type" is not "disabled".
    ///
    /// - parallel_ensemble: whether to compute the ensemble using a thread
    ///     pool. only valid if "ensemble_type" is not "disabled".
    ///
    /// - backshift_parsing: boolean for whether to enable backshift parsing.
    ///     In backshift parsing, whenever we reach a leaf (or a node that has
    ///     been visited too few times), we return to the root of the tree and
    ///     use the most recently-seen symbols to traverse the tree, hopefully
    ///     arriving at a location with a more accurate SPA.
    ///
    /// - backshift_ctx_len: the desired depth to arrive at after backshift
    ///     parsing; i.e., the number of symbols to traverse from the root.
    ///     Only valid if "backshift_parsing" is True.
    ///
    /// - backshift_min_count: if the number of times a node has been
    ///     traversed is less than this, backshift parsing is triggered.
    ///     Only valid if "backshift_parsing" is True.
    ///
    /// - backshift_break_at_phrase: whether to continue backshift parsing
    ///     at a certain shift after a return to the root, or to move on to
    ///     the next shift.
    ///
    /// The default value of the parameters (i.e., if you never previously set
    /// them) is as follows:
    ///     - gamma: 0.5
    ///     - lb: 1e-4
    ///     - temp: 1
    ///     - lb_or_temp_first: lb_first
    ///     - adaptive_gamma: disabled
    ///     - ensemble: disabled
    ///     - backshift_parsing: True
    ///     - backshift_ctx_len: 5
    ///     - backshift_min_count: 1
    ///     - backshift_break_at_phrase: False
    ///
    #[pyo3(signature = (gamma=None, lb=None, temp=None, lb_or_temp_first=None, adaptive_gamma=None,
        ensemble_type=None, ensemble_n=None, parallel_ensemble=None, backshift_parsing=None, backshift_ctx_len=None,
        backshift_min_count=None, backshift_break_at_phrase=None))]
    pub fn set_inference_params(
        &mut self,
        gamma: Option<f64>,
        lb: Option<f64>,
        temp: Option<f64>,
        lb_or_temp_first: Option<&str>,
        adaptive_gamma: Option<&str>,
        ensemble_type: Option<&str>,
        ensemble_n: Option<u32>,
        parallel_ensemble: Option<bool>,
        backshift_parsing: Option<bool>,
        backshift_ctx_len: Option<u64>,
        backshift_min_count: Option<u64>,
        backshift_break_at_phrase: Option<bool>,
    ) -> PyResult<()> {
        set_all_parameters(
            &mut self.params,
            gamma,
            lb,
            temp,
            lb_or_temp_first,
            adaptive_gamma,
            ensemble_type,
            ensemble_n,
            parallel_ensemble,
            backshift_parsing,
            backshift_ctx_len,
            backshift_min_count,
            backshift_break_at_phrase,
        )
    }

    /// Set the parameters used for sequence generation. Note that temperature
    /// and topk are not present here, as they are arguments to the generation
    /// function itself. See self.get_generation_params for the current
    /// parameter values.
    ///
    /// See self.set_inference_params for descriptions of all parameters and
    /// their possible values.
    ///
    /// The default value of the parameters (i.e., if you never previously set
    /// them) is as follows:
    ///     - gamma: 0.5
    ///     - adaptive_gamma: disabled
    ///     - ensemble: disabled
    ///     - backshift_parsing: True
    ///     - backshift_ctx_len: 5
    ///     - backshift_min_count: 1
    ///     - backshift_break_at_phrase
    ///
    #[pyo3(signature = (gamma=None, adaptive_gamma=None, ensemble_type=None, ensemble_n=None,
        parallel_ensemble=None, backshift_parsing=None, backshift_ctx_len=None, backshift_min_count=None,
        backshift_break_at_phrase=None))]
    pub fn set_generation_params(
        &mut self,
        gamma: Option<f64>,
        adaptive_gamma: Option<&str>,
        ensemble_type: Option<&str>,
        ensemble_n: Option<u32>,
        parallel_ensemble: Option<bool>,
        backshift_parsing: Option<bool>,
        backshift_ctx_len: Option<u64>,
        backshift_min_count: Option<u64>,
        backshift_break_at_phrase: Option<bool>,
    ) -> PyResult<()> {
        set_all_parameters(
            &mut self.params,
            gamma,
            None,
            None,
            None,
            adaptive_gamma,
            ensemble_type,
            ensemble_n,
            parallel_ensemble,
            backshift_parsing,
            backshift_ctx_len,
            backshift_min_count,
            backshift_break_at_phrase,
        )
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
                    .train_on_block(u8_sequence, &mut self.params, &mut self.state)?
            }
            SequenceType::Char(character_sequence) => {
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::Char(CharacterSequence::new(
                        &SequenceParams::CharMap(character_sequence.character_map.clone()),
                    )?));
                self.spa
                    .train_on_block(character_sequence, &mut self.params, &mut self.state)?
            }
            SequenceType::U32(u32_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::U32(U32Sequence::new(
                    &SequenceParams::AlphaSize(input.alphabet_size()?),
                )?));
                self.spa
                    .train_on_block(u32_sequence, &mut self.params, &mut self.state)?
            }
        })
    }

    /// Given the SPA that has been trained thus far, compute the cross-entropy
    /// log loss of a test sequence.
    #[pyo3(signature = (input, context=None))]
    pub fn compute_test_loss(
        &mut self,
        input: Sequence,
        context: Option<Sequence>,
    ) -> PyResult<f64> {
        let context = if let Some(ctx) = context {
            ctx.sequence.to_vec()
        } else {
            vec![]
        };
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        Ok(match &input.sequence {
            SequenceType::U8(u8_sequence) => self.spa.test_on_block(
                u8_sequence,
                &mut self.params,
                &mut self.state,
                Some(&context),
            )?,
            SequenceType::Char(character_sequence) => self.spa.test_on_block(
                character_sequence,
                &mut self.params,
                &mut self.state,
                Some(&context),
            )?,
            SequenceType::U32(u32_sequence) => self.spa.test_on_block(
                u32_sequence,
                &mut self.params,
                &mut self.state,
                Some(&context),
            )?,
        })
    }

    #[pyo3(signature = (inputs, contexts=None))]
    pub fn compute_test_loss_parallel(
        &mut self,
        inputs: Vec<Sequence>,
        contexts: Option<Vec<Sequence>>,
    ) -> PyResult<Vec<f64>> {
        let contexts = if let Some(ctx) = contexts {
            ctx.iter().map(|x| x.sequence.to_vec()).collect_vec()
        } else {
            (0..inputs.len()).map(|_| vec![]).collect_vec()
        };

        if self.empty_seq_of_correct_datatype.is_some() {
            for seq in inputs.iter() {
                self.empty_seq_of_correct_datatype
                    .as_ref()
                    .unwrap()
                    .assert_types_match(&seq.sequence)?;
            }
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        let input_ctx_state_params = inputs
            .into_iter()
            .zip(contexts.into_iter())
            .map(|(input, ctx)| (input, ctx, self.state.clone(), self.params.clone()))
            .collect_vec();

        let mut outputs = (0..input_ctx_state_params.len())
            .map(|_| 0f64)
            .collect_vec();
        input_ctx_state_params
            .into_par_iter()
            .map(
                |(input, ctx, mut state, mut params)| match &input.sequence {
                    SequenceType::U8(u8_sequence) => self
                        .spa
                        .test_on_block(u8_sequence, &mut params, &mut state, Some(&ctx))
                        .unwrap(),
                    SequenceType::Char(character_sequence) => self
                        .spa
                        .test_on_block(character_sequence, &mut params, &mut state, Some(&ctx))
                        .unwrap(),
                    SequenceType::U32(u32_sequence) => self
                        .spa
                        .test_on_block(u32_sequence, &mut params, &mut state, Some(&ctx))
                        .unwrap(),
                },
            )
            .collect_into_vec(&mut outputs);

        Ok(outputs)
    }

    /// Computes the SPA for every symbol in the alphabet, using the LZ78
    /// context reached at the end of parsing the last training block
    pub fn compute_spa_at_current_state(&mut self) -> PyResult<Vec<f64>> {
        Ok(self
            .spa
            .spa(&mut self.params, &mut self.state, None)?
            .to_vec())
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
    ///
    /// Returns a tuple of the generated sequence and that sequence's log loss,
    /// or perplexity.
    ///
    /// Errors if the SPA has not been trained so far, or if the seed data is
    /// not over the same alphabet as the training data.
    #[pyo3(signature = (len, seed_data=None, temperature=0.5, top_k=5))]
    pub fn generate_data(
        &mut self,
        len: u64,
        seed_data: Option<Sequence>,
        temperature: f64,
        top_k: u32,
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

        let loss = match seed_data {
            None => match &mut output {
                SequenceType::U8(u8_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &mut self.params,
                    temperature,
                    Some(top_k),
                    None,
                    u8_sequence,
                )?,
                SequenceType::Char(character_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &mut self.params,
                    temperature,
                    Some(top_k),
                    None,
                    character_sequence,
                )?,
                SequenceType::U32(u32_sequence) => generate_sequence(
                    &mut self.spa,
                    len,
                    &mut self.params,
                    temperature,
                    Some(top_k),
                    None,
                    u32_sequence,
                )?,
            },
            Some(seed_data) => match (&mut output, &seed_data.sequence) {
                (SequenceType::U8(output_seq), SequenceType::U8(seed_seq)) => generate_sequence(
                    &mut self.spa,
                    len,
                    &mut self.params,
                    temperature,
                    Some(top_k),
                    Some(seed_seq),
                    output_seq,
                )?,
                (SequenceType::Char(output_seq), SequenceType::Char(seed_seq)) => {
                    generate_sequence(
                        &mut self.spa,
                        len,
                        &mut self.params,
                        temperature,
                        Some(top_k),
                        Some(seed_seq),
                        output_seq,
                    )?
                }
                (SequenceType::U32(output_seq), SequenceType::U32(seed_seq)) => generate_sequence(
                    &mut self.spa,
                    len,
                    &mut self.params,
                    temperature,
                    Some(top_k),
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
        bytes.extend(self.gen_params.to_bytes()?);
        bytes.extend(self.state.to_bytes()?);
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Extracts information about the depth of leaves of the LZ78 prefix tree
    /// underlying this SPA.
    pub fn get_debug_info(&self) -> PyResult<()> {
        if let SPAParams::LZ78(params) = &self.params {
            if !params.debug {
                return Err(PyAssertionError::new_err(
                    "Must instantiate an LZ78 SPA with debug=true to get debug info",
                ));
            }
        } else {
            unreachable!();
        }

        todo!();
        // Ok(LZ78DebugInfo {
        //     debug_info: self.spa.get_debug_info().clone(),
        // })
    }

    /// Resets the running list of node depths (i.e., the one returned by
    /// LZ78DebugInfo.get_depths_traversed)
    pub fn clear_debug_depths(&mut self) {
        todo!()
        // self.spa.debug.clear_depths_traversed();
    }

    /// Prunes the nodes of the tree that have been visited fewer than a
    /// certain number of times
    pub fn prune(&mut self, min_count: u64) {
        self.spa.prune(min_count);
    }

    pub fn shrink_to_fit(&mut self) {
        self.spa.shrink_to_fit();
    }
}

#[pyfunction]
pub fn spa_from_file(filename: &str) -> PyResult<LZ78SPA> {
    let mut file = File::open(filename)?;
    let mut buf: Vec<u8> = Vec::new();
    file.read_to_end(&mut buf)?;
    let mut bytes: Bytes = buf.into();
    spa_from_bytes_helper(&mut bytes)
}

#[pyfunction]
/// Constructs a trained SPA from its byte array representation.
pub fn spa_from_bytes<'py>(bytes: Py<PyBytes>, py: Python<'py>) -> PyResult<LZ78SPA> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    spa_from_bytes_helper(&mut bytes)
}

fn spa_from_bytes_helper(bytes: &mut Bytes) -> PyResult<LZ78SPA> {
    let spa: RustLZ78SPA<DirichletSPATree> = RustLZ78SPA::from_bytes(bytes)?;
    let empty_seq_of_correct_datatype = match bytes.get_u8() {
        0 => Some(SequenceType::from_bytes(bytes)?),
        1 => None,
        _ => {
            return Err(PyAssertionError::new_err(
                "Error reading encoded sequence from bytes",
            ))
        }
    };
    let alphabet_size = bytes.get_u32_le();
    let params = SPAParams::from_bytes(bytes)?;
    let gen_params = SPAParams::from_bytes(bytes)?;
    let state = SPAState::from_bytes(bytes)?;

    Ok(LZ78SPA {
        spa,
        alphabet_size,
        empty_seq_of_correct_datatype,
        params,
        gen_params,
        state,
    })
}

// /// Debugging information for the LZ78SPA; i.e., the depths of the leaves.
// /// This class cannot be instantiated, but rather is produced via
// /// `LZ78SPA.get_debug_info()`
// #[pyclass]
// pub struct LZ78DebugInfo {
//     debug_info: LZ78DebugState,
// }

// #[pymethods]
// impl LZ78DebugInfo {
//     /// Get the length of the longest branch of the LZ78 prefix tree
//     fn get_max_leaf_depth(&self) -> u32 {
//         self.debug_info.max_depth
//     }

//     /// Get the length of the shortest branch of the LZ78 prefix tree
//     fn get_min_leaf_depth(&self) -> u32 {
//         *self.debug_info.leaf_depths.values().min().unwrap_or(&0)
//     }

//     /// Get the (unweighted) average depth of all leaves of the LZ78 prefix tree
//     fn get_mean_leaf_depth(&self) -> f64 {
//         self.debug_info
//             .leaf_depths
//             .values()
//             .map(|&x| x as f64)
//             .sum::<f64>()
//             / self.debug_info.leaf_depths.len() as f64
//     }

//     /// Returns the depth of all leaves of the LZ78 tree as a list (in no
//     /// particular order)
//     fn get_leaf_depths(&self) -> Vec<u32> {
//         self.debug_info
//             .leaf_depths
//             .values()
//             .map(|&x| x)
//             .collect_vec()
//     }

//     /// Returns the longest LZ78 phrase encoded by the prefix tree,
//     /// as a list of integer symbols
//     fn get_longest_branch(&self) -> Vec<u32> {
//         self.debug_info.get_longest_branch()
//     }

//     /// Returns the depth of the currently-traversed node in the LZ tree,
//     /// for each timepoint. Does not distinguish between training, inference,
//     /// and generation; it is recommended to use `spa.clear_debug_depths()`
//     /// between training and inference or generation.
//     fn get_depths_traversed(&self) -> Vec<u32> {
//         self.debug_info.depths_traversed.clone()
//     }
// }
