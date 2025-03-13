use crate::{spa::util::LbAndTemp, storage::ToFromBytes};

use super::{
    config::{DirichletConfig, SPAConfig},
    generation::{gen_symbol_from_spa, GenerationSPA, GenerationSPATree},
    states::SPAState,
    util::apply_lb_and_temp_to_spa,
    InfOutOptions, InferenceOutput, LZWTree, SPATree, SPA,
};
use anyhow::Result;
use bytes::Bytes;
use hashbrown::{HashMap, HashSet};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct DirichletSPATree {
    pub ns: Vec<u64>,
    pub branches: LZWTree,
}

impl ToFromBytes for DirichletSPATree {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.ns.to_bytes()?;
        bytes.extend(self.branches.to_bytes()?);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let ns = Vec::<u64>::from_bytes(bytes)?;
        let branches = LZWTree::from_bytes(bytes)?;
        Ok(Self { ns, branches })
    }
}

impl DirichletSPATree {
    fn spa_for_symbol_basic(
        &self,
        idx: u64,
        sym: u32,
        dirichlet_config: &DirichletConfig,
    ) -> Result<f64> {
        let count = match self.branches.get_child_idx(idx, sym) {
            Some(i) => self.ns[*i as usize] + 1,
            None => 0,
        } as f64;
        Ok((count + dirichlet_config.gamma)
            / (self.ns[idx as usize] as f64
                + dirichlet_config.gamma * dirichlet_config.alphabet_size as f64))
    }
}

impl SPATree for DirichletSPATree {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        _state: &mut SPAState,
    ) -> Result<f64> {
        let loss = if config.compute_training_loss() {
            -self
                .spa_for_symbol_basic(idx, sym, config.try_get_dirichlet_mut()?)?
                .log2()
        } else {
            0.0
        };
        self.ns[idx as usize] += 1;
        Ok(loss)
    }

    fn spa_for_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let dirichlet_config = config.try_get_dirichlet_mut()?;
        if dirichlet_config.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(idx, config, state, context_syms)?[sym as usize]);
        }
        self.spa_for_symbol_basic(idx, sym, &dirichlet_config)
    }

    fn spa(
        &self,
        idx: u64,
        config: &mut SPAConfig,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let config = config.try_get_dirichlet_mut()?;

        let mut spa = Array1::zeros(config.alphabet_size as usize);
        for sym in 0..config.alphabet_size {
            if let Some(i) = self.get_child_idx(idx, sym) {
                spa[sym as usize] = self.ns[*i as usize] as f64 + 1.0;
            }
        }

        spa = (spa + config.gamma)
            / (self.ns[idx as usize] as f64 + config.gamma * config.alphabet_size as f64);
        apply_lb_and_temp_to_spa(&mut spa, config.lb_and_temp, None);
        Ok(spa)
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-self
            .spa_for_symbol(idx, sym, config, state, context_syms)?
            .log2())
    }

    fn add_new(&mut self, _config: &SPAConfig, parent_idx: u64, sym: u32) -> Result<()> {
        self.branches
            .add_leaf(parent_idx, sym, self.ns.len() as u64);
        self.ns.push(0);

        Ok(())
    }

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        self.branches.get_child_idx(idx, sym)
    }

    fn num_symbols_seen(&self, idx: u64) -> u64 {
        self.ns[idx as usize]
    }

    fn new(_config: &SPAConfig) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            ns: vec![0],
            branches: LZWTree::new(),
        })
    }

    fn prune(&mut self, min_count: u64) {
        let mut remove = HashSet::new();
        let mut replace = HashMap::new();
        let mut write_idx = 0;
        for i in 0..self.ns.len() {
            if self.ns[i] < min_count {
                remove.insert(i as u64);
            } else {
                self.ns[write_idx] = self.ns[i];
                replace.insert(i as u64, write_idx as u64);
                write_idx += 1;
            }
        }
        self.ns.truncate(write_idx.max(1));
        self.branches.remove_batch(&remove);
        self.branches.replace(&replace);
    }

    fn shrink_to_fit(&mut self) {
        self.branches.shrink_to_fit();
        self.ns.shrink_to_fit();
    }
}

impl GenerationSPATree for DirichletSPATree {
    fn input_seed_data_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.test_on_symbol(idx, sym, config, state, None)
    }

    fn generate_one_symbol(
        &self,
        idx: u64,
        rng_sample: f64,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(
            rng_sample,
            &self.spa(idx, config, state, Some(context_syms))?,
            temperature,
            topk,
        )
    }
}
pub struct DirichletSPA {
    pub n: u64,
    pub counts: Array1<f64>,
}

impl SPA for DirichletSPA {
    fn train_on_symbol(
        &mut self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        let loss = -self.spa_for_symbol(sym, config, state, None)?.log2();
        self.counts[sym as usize] += 1.0;
        self.n += 1;
        Ok(loss)
    }

    fn spa(
        &self,
        config: &mut SPAConfig,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let config = config.try_get_dirichlet_mut()?;
        let mut spa = self.counts.clone() * config.gamma
            / (self.n as f64 + config.gamma * config.alphabet_size as f64);
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
        let dirichlet_config = config.try_get_dirichlet_mut()?;
        if dirichlet_config.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(config, state, context_syms)?[sym as usize]);
        }
        Ok((self.counts[sym as usize] + dirichlet_config.gamma)
            / (self.n as f64 + dirichlet_config.gamma * dirichlet_config.alphabet_size as f64))
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
    ) -> Result<InferenceOutput> {
        if inf_out_options.output_probs() {
            let dist = self.spa(config, state, context_syms)?;
            let loss = -dist[sym as usize].log2();
            let ppl = loss.exp2();
            return Ok(InferenceOutput::new(
                loss,
                ppl,
                vec![loss],
                vec![dist.to_vec()],
            ));
        }

        let loss = -self
            .spa_for_symbol(sym, config, state, context_syms)?
            .log2();
        let ppl = loss.exp2();
        let losses = if inf_out_options.output_losses() {
            vec![loss]
        } else {
            vec![]
        };

        Ok(InferenceOutput::new(loss, ppl, losses, vec![]))
    }

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            n: 0,
            counts: Array1::zeros(config.alphabet_size() as usize),
        })
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl GenerationSPA for DirichletSPA {
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        Ok(self
            .test_on_symbol(sym, config, state, InfOutOptions::Basic, None)?
            .avg_log_loss)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(
            rng_sample,
            &self.spa(config, state, Some(context_syms))?,
            temperature,
            topk,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::spa::config::DirichletConfigBuilder;

    use super::*;

    #[test]
    fn test_dirichlet_tree_to_from_bytes() {
        let mut config = DirichletConfigBuilder::new(3)
            .gamma(0.2)
            .lb_and_temp(0.1, 2.0, true)
            .build_enum();
        let mut state = SPAState::None;
        let mut spa = DirichletSPATree::new(&config).expect("failed to make DirichletSPA");
        spa.train_spa_on_symbol(0, 0, &mut config, &mut state)
            .expect("train dirichlet spa failed");
        spa.train_spa_on_symbol(0, 2, &mut config, &mut state)
            .expect("train dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPATree::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.ns, new_spa.ns);
    }
}
