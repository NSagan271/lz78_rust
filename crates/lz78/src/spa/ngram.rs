use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::{array, Array1, Array2, Axis};

use crate::storage::ToFromBytes;

use super::{
    config::{NGramConfig, SPAConfig},
    states::{NGramMixtureState, SPAState},
    util::{apply_lb_and_temp_to_spa, LbAndTemp},
    InfOutOptions, InferenceOutput, SPA,
};

fn get_depth_spa_weights(min_n: f64, max_n: f64) -> Array1<f64> {
    let norm_dep = (Array1::range(min_n, max_n + 1.0, 1.0) - min_n) / (max_n - min_n + 1e-6);
    let mut weights = norm_dep.exp();
    weights /= weights.sum();
    weights
}

fn get_average_spa_weights(min_n: u8, max_n: u8) -> Array1<f64> {
    Array1::ones((max_n - min_n + 1) as usize) / (max_n - min_n + 1) as f64
}

#[derive(Debug, Clone)]
pub struct NGramSPA {
    pub counts: Vec<HashMap<u64, u64>>,
    num_sym: u64,
}

impl NGramSPA {
    fn spa_for_symbol_single_n(
        &self,
        alphabet_size: u32,
        n: u8,
        gamma: f64,
        state: &NGramMixtureState,
        state_with_sym_added: &NGramMixtureState,
    ) -> f64 {
        let denom = *self.counts[n as usize]
            .get(&state.get_encoded_len_n_ctx(n, alphabet_size))
            .unwrap_or(&0) as f64;
        let numer = *self.counts[n as usize + 1]
            .get(&state_with_sym_added.get_encoded_len_n_ctx(n + 1, alphabet_size))
            .unwrap_or(&0) as f64;
        (numer + gamma) / (denom + (alphabet_size as f64) * gamma)
    }

    fn spa_for_symbol_basic(
        &self,
        sym: u32,
        config: &NGramConfig,
        state: &NGramMixtureState,
    ) -> Result<f64> {
        if state.context_len < config.min_n {
            return Ok(1.0 / config.alphabet_size as f64);
        }

        let mut clone_state = state.clone();
        clone_state.add_sym(sym, config.alphabet_size, config.max_n);

        let max_n = state.context_len.min(config.max_n);
        let weights = match config.ensemble {
            super::config::Ensemble::Average(_) => get_average_spa_weights(config.min_n, max_n),
            super::config::Ensemble::Entropy(_) => {
                bail!("Entropy ensemble cannot be computed with spa_for_symbol_basic")
            }
            super::config::Ensemble::Depth(_) => {
                get_depth_spa_weights(config.min_n as f64, max_n as f64)
            }
            super::config::Ensemble::None => array![1.0],
        };

        let mut spa_vals = Array1::zeros((max_n - config.min_n + 1) as usize);
        for i in config.min_n..=max_n {
            spa_vals[(i - config.min_n) as usize] = self.spa_for_symbol_single_n(
                config.alphabet_size,
                i,
                config.gamma,
                state,
                &clone_state,
            );
        }

        Ok((spa_vals * weights).sum())
    }

    fn maybe_add_ctx_to_state(
        &self,
        state: &mut NGramMixtureState,
        config: &NGramConfig,
        context_syms: Option<&[u32]>,
    ) {
        if let Some(syms) = context_syms {
            if state.context_len == 0 {
                for sym in syms {
                    state.add_sym(*sym, config.alphabet_size, config.max_n);
                }
            }
        }
    }
}

impl SPA for NGramSPA {
    fn train_on_symbol(
        &mut self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.num_sym += 1;

        let state = state.try_get_ngram()?;
        let config = config.try_get_ngram()?;
        state.add_sym(sym, config.alphabet_size, config.max_n);

        if state.context_len >= config.min_n {
            for i in config.min_n..state.context_len {
                let id = state.get_encoded_len_n_ctx(i, config.alphabet_size);
                let new_count = self.counts[i as usize].get(&id).unwrap_or(&0) + 1;
                self.counts[i as usize].insert(id, new_count);
            }
        }

        Ok(0.0)
    }

    fn spa(
        &self,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let state = state.try_get_ngram()?;
        let config = config.try_get_ngram()?;

        self.maybe_add_ctx_to_state(state, config, context_syms);

        if state.context_len < config.min_n {
            return Ok(Array1::ones(config.alphabet_size as usize) / config.alphabet_size as f64);
        }

        let mut spa = if !config.ensemble.is_entropy() {
            let mut spa = Array1::zeros(config.alphabet_size as usize);

            for sym in 0..config.alphabet_size {
                spa[sym as usize] = self.spa_for_symbol_basic(sym, config, state)?;
            }
            spa
        } else {
            let max_n = state.context_len.min(config.max_n);
            let mut spas = Array2::zeros((
                max_n as usize - config.min_n as usize + 1,
                config.alphabet_size as usize,
            ));
            for sym in 0..config.alphabet_size {
                let mut clone_state = state.clone();
                clone_state.add_sym(sym, config.alphabet_size, config.max_n);

                for n in config.min_n..=max_n {
                    spas[(n as usize, sym as usize)] = self.spa_for_symbol_single_n(
                        config.alphabet_size,
                        n,
                        config.gamma,
                        state,
                        &clone_state,
                    );
                }
            }

            let entropy = -(spas.clone() * spas.clone().log2()).sum_axis(Axis(1));
            let ent_min = *entropy.iter().min_by(|&x, &y| x.total_cmp(y)).unwrap();
            let ent_max = *entropy.iter().max_by(|&x, &y| x.total_cmp(y)).unwrap();
            let norm_ent = (entropy - ent_min) / (ent_max - ent_min + 1e-10);
            let mut weights = (-norm_ent / 2.0).exp();
            weights /= weights.sum();

            (spas.reversed_axes() * weights)
                .reversed_axes()
                .sum_axis(Axis(0))
        };

        apply_lb_and_temp_to_spa(&mut spa, config.lb_and_temp, None);

        Ok(spa)
    }

    fn spa_for_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let ngram_state = state.try_get_ngram()?;
        let ngram_config = config.try_get_ngram()?;
        self.maybe_add_ctx_to_state(ngram_state, ngram_config, context_syms);

        if ngram_config.ensemble.is_entropy() || ngram_config.lb_and_temp != LbAndTemp::Skip {
            Ok(self.spa(config, state, context_syms)?[sym as usize])
        } else {
            self.spa_for_symbol_basic(sym, ngram_config, ngram_state)
        }
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
    ) -> Result<InferenceOutput> {
        let res = if inf_out_options.output_probs() {
            let spa = self.spa(config, state, context_syms)?;
            let loss = -spa[sym as usize].log2();
            InferenceOutput::new(loss, loss.exp2(), vec![loss], vec![spa.to_vec()])
        } else {
            let loss = -self
                .spa_for_symbol(sym, config, state, context_syms)?
                .log2();
            let ppl = loss.exp2();
            let losses = if inf_out_options.output_losses() {
                vec![loss]
            } else {
                vec![]
            };
            InferenceOutput::new(loss, ppl, losses, vec![])
        };
        let config = config.try_get_ngram()?;
        state
            .try_get_ngram()?
            .add_sym(sym, config.alphabet_size, config.max_n);

        Ok(res)
    }

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let config = config.try_get_ngram()?;
        Ok(Self {
            counts: (0..config.max_n + 2).map(|_| HashMap::new()).collect_vec(),
            num_sym: 0,
        })
    }

    fn num_symbols_seen(&self) -> u64 {
        self.num_sym
    }
}

impl ToFromBytes for NGramSPA {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.counts.len() as u64);
        for x in self.counts.iter() {
            bytes.extend(x.to_bytes()?);
        }
        bytes.put_u64_le(self.num_sym);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le() as usize;
        let mut counts = Vec::with_capacity(n);
        for _ in 0..n {
            counts.push(HashMap::<u64, u64>::from_bytes(bytes)?);
        }
        let num_sym = bytes.get_u64_le();
        Ok(Self { counts, num_sym })
    }
}

#[cfg(test)]
mod tests {
    use crate::{sequence::BinarySequence, spa::config::NGramConfigBuilder};
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut config = NGramConfigBuilder::new(2, 4).build_enum();
        let mut state = config.get_new_state();
        let mut spa: NGramSPA = NGramSPA::new(&config).expect("failed to make NGram SPA");
        spa.train_on_block(&input, &mut config, &mut state)
            .expect("failed to train spa");

        state.reset();
        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        state.reset();
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }
}
