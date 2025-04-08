use itertools::Itertools;
use lz78::sequence::{
    CharacterSequence, Sequence as RustSequence, SequenceConfig, U32Sequence, U8Sequence,
};
use lz78::spa::config::{Ensemble, NGramConfigBuilder};
use lz78::spa::{config::SPAConfig, ngram::NGramSPA as RustNGramSPA, states::SPAState};
use lz78::spa::{InfOutOptions, SPA};
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::{IntoPyDict, PyDict};

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
            .ensemble(Ensemble::Average(ensemble_size as u32))
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
            )?,
            SequenceType::Char(character_sequence) => self.spa.test_on_block(
                character_sequence,
                &mut self.config,
                &mut inf_state,
                inf_options,
                Some(&context),
            )?,
            SequenceType::U32(u32_sequence) => self.spa.test_on_block(
                u32_sequence,
                &mut self.config,
                &mut inf_state,
                inf_options,
                Some(&context),
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
}
