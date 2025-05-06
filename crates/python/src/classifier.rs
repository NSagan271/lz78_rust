use std::{fs::File, io::Read};

use bytes::Bytes;
use itertools::Itertools;
use lz78::spa::{InfOutOptions, SPA};
use ndarray::Array2;
use pyo3::{pyclass, pyfunction, pymethods, PyResult};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    sequence::{Sequence, SequenceType},
    spa::{spa_from_bytes_helper, LZ78SPA},
};

#[pyclass]
pub struct LZ78Classifier {
    spas: Vec<LZ78SPA>,
}

#[pymethods]
impl LZ78Classifier {
    #[new]
    pub fn new() -> Self {
        Self { spas: Vec::new() }
    }

    pub fn add_spa(&mut self, spa: &LZ78SPA) {
        self.spas.push(spa.clone());
    }

    #[pyo3(signature = (gamma=None, lb=None, temp=None, lb_or_temp_first=None, adaptive_gamma=None,
        ensemble_type=None, ensemble_n=None, backshift_parsing=None, backshift_ctx_len=None,
        backshift_break_at_phrase=None))]
    pub fn set_inference_config(
        &mut self,
        gamma: Option<f64>,
        lb: Option<f64>,
        temp: Option<f64>,
        lb_or_temp_first: Option<&str>,
        adaptive_gamma: Option<&str>,
        ensemble_type: Option<&str>,
        ensemble_n: Option<u32>,
        backshift_parsing: Option<bool>,
        backshift_ctx_len: Option<u64>,
        backshift_break_at_phrase: Option<bool>,
    ) -> PyResult<()> {
        for spa in self.spas.iter_mut() {
            spa.set_inference_config(
                gamma,
                lb,
                temp,
                lb_or_temp_first,
                adaptive_gamma,
                ensemble_type,
                ensemble_n,
                backshift_parsing,
                backshift_ctx_len,
                backshift_break_at_phrase,
            )?;
        }
        Ok(())
    }

    pub fn classify(&mut self, input: Sequence) -> PyResult<usize> {
        Ok(self
            .spas
            .par_iter_mut()
            .enumerate()
            .map(|(i, spa)| {
                let mut state = spa.state.clone();
                state.reset();
                (
                    (match &input.sequence {
                        SequenceType::U8(u8_sequence) => spa
                            .spa
                            .test_on_block(
                                u8_sequence,
                                &mut spa.config,
                                &mut state,
                                InfOutOptions::Basic,
                                None,
                                None,
                            )
                            .unwrap(),
                        SequenceType::Char(character_sequence) => spa
                            .spa
                            .test_on_block(
                                character_sequence,
                                &mut spa.config,
                                &mut state,
                                InfOutOptions::Basic,
                                None,
                                None,
                            )
                            .unwrap(),
                        SequenceType::U32(u32_sequence) => spa
                            .spa
                            .test_on_block(
                                u32_sequence,
                                &mut spa.config,
                                &mut state,
                                InfOutOptions::Basic,
                                None,
                                None,
                            )
                            .unwrap(),
                    })
                    .avg_log_loss,
                    i,
                )
            })
            .min_by(|x, y| {
                if x.0 == y.0 {
                    y.1.cmp(&x.1)
                } else {
                    x.0.total_cmp(&y.0)
                }
            })
            .unwrap_or((0.0, 0))
            .1)
    }

    pub fn classify_batch(
        &mut self,
        inputs: Vec<Sequence>,
        num_threads: usize,
    ) -> PyResult<Vec<usize>> {
        let mut losses: Array2<f32> = Array2::zeros((self.spas.len(), inputs.len()));
        for (i, spa) in self.spas.iter_mut().enumerate() {
            spa.compute_test_loss_parallel_helper(
                &inputs,
                (0..inputs.len()).map(|_| Vec::new()).collect_vec(),
                num_threads,
                InfOutOptions::Basic,
                false,
                None,
            )?
            .iter()
            .enumerate()
            .for_each(|(j, x)| losses[(i, j)] = x.0.avg_log_loss);
        }
        Ok(losses
            .columns()
            .into_iter()
            .map(|col| {
                col.iter()
                    .enumerate()
                    .min_by(|x, y| x.1.total_cmp(y.1))
                    .unwrap()
                    .0
            })
            .collect_vec())
    }
}

#[pyfunction]
pub fn classifier_from_files(filenames: Vec<String>) -> PyResult<LZ78Classifier> {
    let mut spas = Vec::new();
    for filename in filenames {
        let mut file = File::open(filename)?;
        let mut buf: Vec<u8> = Vec::new();
        file.read_to_end(&mut buf)?;
        let mut bytes: Bytes = buf.into();
        spas.push(spa_from_bytes_helper(&mut bytes)?);
    }

    Ok(LZ78Classifier { spas })
}
