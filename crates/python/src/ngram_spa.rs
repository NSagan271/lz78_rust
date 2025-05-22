use std::fs::File;
use std::io::{Read, Write};

use crate::spa::compute_test_loss_parallel_on_spa;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use lz78::sequence::{
    CharacterSequence, Sequence as RustSequence, SequenceConfig, U32Sequence, U8Sequence,
};
use lz78::spa::config::{Ensemble, NGramConfigBuilder};
use lz78::spa::{config::SPAConfig, ngram::NGramSPA as RustNGramSPA, states::SPAState};
use lz78::spa::{InfOutOptions, SPA};
use lz78::storage::ToFromBytes;
use numpy::{Ix3, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};

use crate::sequence::{Sequence, SequenceType};

#[pyclass]
#[derive(Clone)]
pub struct NGramSPA {
    pub spa: RustNGramSPA,
    alphabet_size: u32,
    empty_seq_of_correct_datatype: Option<SequenceType>,
    pub config: SPAConfig,
    pub state: SPAState,
}

#[pymethods]
impl NGramSPA {
    #[new]
    #[pyo3(signature = (alphabet_size, n, gamma=0.5, ensemble_size=1))]
    pub fn new(alphabet_size: u32, n: u8, gamma: f64, ensemble_size: u8) -> PyResult<Self> {
        let config = NGramConfigBuilder::new(alphabet_size, n)
            .gamma(gamma)
            .ensemble(Ensemble::Depth(ensemble_size as u32))
            .build_enum();
        Ok(Self {
            spa: RustNGramSPA::new(&config)?,
            alphabet_size,
            empty_seq_of_correct_datatype: None,
            state: config.get_new_state(),
            config,
        })
    }

    /// Reset the ngram SPA context
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    /// Sets the dirichlet smoothing parameter and ensemble type.
    ///
    /// Inputs:
    /// - gamma: dirichlet smoothing parameter (positive float)
    /// - ensemble_type:for ngram ensembles only; either "average", "entropy",
    ///     or "depth".
    ///
    #[pyo3(signature = (gamma=None, ensemble_type=None))]
    pub fn set_inference_params(
        &mut self,
        gamma: Option<f64>,
        ensemble_type: Option<&str>,
    ) -> PyResult<()> {
        if let Some(g) = gamma {
            self.config.try_get_ngram_mut()?.gamma = g;
        }

        let curr_ens_n = self.config.try_get_ngram()?.ensemble.get_num_states() as u32;
        if let Some(ens) = ensemble_type {
            self.config.try_get_ngram_mut()?.ensemble = match ens.to_lowercase().as_str() {
                "average" => Ensemble::Average(curr_ens_n),
                "entropy" => Ensemble::Entropy(curr_ens_n),
                "depth" => Ensemble::Depth(curr_ens_n),
                _ => {
                    return Err(PyAssertionError::new_err("Unexpected value of esemble"));
                }
            };
        }

        Ok(())
    }

    pub fn get_inference_config<'py>(&self) -> PyResult<Bound<'py, PyDict>> {
        let config = self.config.try_get_ngram()?;

        todo!()
    }

    #[pyo3(signature = (input))]
    pub fn train_on_block<'py>(&mut self, input: Sequence) -> PyResult<()> {
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

        match &input.sequence {
            SequenceType::U8(u8_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::U8(U8Sequence::new(
                    &SequenceConfig::AlphaSize(input.alphabet_size()?),
                )?));
                self.spa
                    .train_on_block(u8_sequence, &mut self.config, &mut self.state)?
            }
            SequenceType::Char(character_sequence) => {
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::Char(CharacterSequence::new(
                        &SequenceConfig::CharMap(character_sequence.character_map.clone()),
                    )?));
                self.spa
                    .train_on_block(character_sequence, &mut self.config, &mut self.state)?
            }
            SequenceType::U32(u32_sequence) => {
                self.empty_seq_of_correct_datatype = Some(SequenceType::U32(U32Sequence::new(
                    &SequenceConfig::AlphaSize(input.alphabet_size()?),
                )?));
                self.spa
                    .train_on_block(u32_sequence, &mut self.config, &mut self.state)?
            }
        };

        Ok(())
    }

    #[pyo3(signature = (input, context=None, output_per_symbol_losses=false, output_prob_dists=false))]
    pub fn compute_test_loss<'py>(
        &mut self,
        py: Python<'py>,
        input: Sequence,
        context: Option<Sequence>,
        output_per_symbol_losses: bool,
        output_prob_dists: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut inf_state = self.state.clone();
        inf_state.reset();

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

        let inf_options = InfOutOptions::from_bools(output_per_symbol_losses, output_prob_dists);

        let res = match &input.sequence {
            SequenceType::U8(u8_sequence) => self.spa.test_on_block(
                u8_sequence,
                &mut self.config,
                &mut inf_state,
                inf_options,
                Some(&context),
                None,
            )?,
            SequenceType::Char(character_sequence) => self.spa.test_on_block(
                character_sequence,
                &mut self.config,
                &mut inf_state,
                inf_options,
                Some(&context),
                None,
            )?,
            SequenceType::U32(u32_sequence) => self.spa.test_on_block(
                u32_sequence,
                &mut self.config,
                &mut inf_state,
                inf_options,
                Some(&context),
                None,
            )?,
        };

        let key_vals: Vec<(&str, PyObject)> = vec![
            ("avg_log_loss", res.avg_log_loss.to_object(py)),
            ("avg_perplexity", res.avg_perplexity.to_object(py)),
            ("log_losses", res.log_losses.to_object(py)),
            ("prob_dists", res.prob_dists.to_object(py)),
        ];

        Ok(key_vals.into_py_dict_bound(py))
    }

    #[pyo3(signature = (inputs, contexts=None, num_threads=16,
        output_per_symbol_losses=false,
        output_prob_dists=false, prob_dist_output=None))]
    pub fn compute_test_loss_parallel<'py>(
        &mut self,
        py: Python<'py>,
        inputs: Vec<Sequence>,
        contexts: Option<Vec<Sequence>>,
        num_threads: usize,
        output_per_symbol_losses: bool,
        output_prob_dists: bool,
        prob_dist_output: Option<&Bound<'py, PyArrayDyn<f32>>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let contexts = if let Some(ctx) = contexts {
            ctx.iter().map(|x| x.sequence.to_vec()).collect_vec()
        } else {
            (0..inputs.len()).map(|_| vec![]).collect_vec()
        };
        let inf_options = InfOutOptions::from_bools(output_per_symbol_losses, output_prob_dists);

        let outputs = if let Some(prob_dist) = prob_dist_output {
            let prob_dist = unsafe { prob_dist.as_array_mut() }
                .into_dimensionality::<Ix3>()
                .unwrap();
            compute_test_loss_parallel_on_spa(
                &self.spa,
                &mut self.config,
                &mut self.state.clone(),
                &self.empty_seq_of_correct_datatype,
                &inputs,
                contexts,
                num_threads,
                inf_options,
                false,
                Some(prob_dist),
            )?
        } else {
            compute_test_loss_parallel_on_spa(
                &self.spa,
                &mut self.config,
                &mut self.state.clone(),
                &self.empty_seq_of_correct_datatype,
                &inputs,
                contexts,
                num_threads,
                inf_options,
                false,
                None,
            )?
        };

        let mut final_outputs = vec![];
        for (res, _) in outputs {
            let key_vals: Vec<(&str, PyObject)> = vec![
                ("avg_log_loss", res.avg_log_loss.to_object(py)),
                ("avg_perplexity", res.avg_perplexity.to_object(py)),
                ("log_losses", res.log_losses.to_object(py)),
                ("prob_dists", res.prob_dists.to_object(py)),
            ];
            final_outputs.push(key_vals.into_py_dict_bound(py));
        }

        Ok(final_outputs)
    }

    fn get_counts_for_context(&self, context: Sequence) -> PyResult<Vec<u64>> {
        let mut state = self.config.get_new_state();
        let config = self.config.try_get_ngram()?;
        let ngram_state = state.try_get_ngram()?;
        for sym in context.sequence.to_vec() {
            ngram_state.add_sym(sym, config.alphabet_size, config.max_n);
        }

        Ok((0..config.alphabet_size)
            .map(|sym| {
                let mut clone_state = ngram_state.clone();
                clone_state.add_sym(sym, config.alphabet_size, config.max_n);
                *self.spa.counts[ngram_state.context_len as usize]
                    .get(
                        &clone_state
                            .get_encoded_len_n_ctx(ngram_state.context_len, config.alphabet_size),
                    )
                    .unwrap_or(&0)
            })
            .collect_vec())
    }

    #[pyo3(signature = (normalized_counts=false))]
    fn to_vec(&self, normalized_counts: bool) -> PyResult<Vec<f32>> {
        Ok(self
            .spa
            .to_vec(self.config.try_get_ngram()?, normalized_counts))
    }

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
        bytes.extend(self.config.to_bytes()?);
        bytes.extend(self.state.to_bytes()?);
        Ok(PyBytes::new_bound(py, &bytes))
    }

    pub fn to_file(&self, filename: &str) -> PyResult<()> {
        let mut bytes = self.spa.to_bytes()?;
        match &self.empty_seq_of_correct_datatype {
            Some(seq) => {
                bytes.put_u8(0);
                bytes.extend(seq.to_bytes()?);
            }
            None => bytes.put_u8(1),
        };
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.config.to_bytes()?);
        bytes.extend(self.state.to_bytes()?);

        let mut file = File::create(filename)?;
        file.write_all(&bytes)?;

        Ok(())
    }

    pub fn get_total_counts(&self) -> u64 {
        self.spa.num_symbols_seen()
    }

    pub fn get_total_nodes(&self) -> u64 {
        self.spa.counts.iter().map(|x| x.len()).sum::<usize>() as u64
    }
}

#[pyfunction]
pub fn ngram_from_file(filename: &str) -> PyResult<NGramSPA> {
    let mut file = File::open(filename)?;
    let mut buf: Vec<u8> = Vec::new();
    file.read_to_end(&mut buf)?;
    let mut bytes: Bytes = buf.into();

    let spa: RustNGramSPA = RustNGramSPA::from_bytes(&mut bytes)?;
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
    let config = SPAConfig::from_bytes(&mut bytes)?;
    let state = SPAState::from_bytes(&mut bytes)?;

    Ok(NGramSPA {
        spa,
        alphabet_size,
        empty_seq_of_correct_datatype,
        config,
        state,
    })
}
