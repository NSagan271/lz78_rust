use crate::{spa::util::LbAndTemp, storage::ToFromBytes};

use super::{
    config::{DirichletConfig, SPAConfig},
    generation::{gen_symbol_from_spa, GenerationSPA, GenerationSPATree},
    lzw_tree::LZWTree,
    states::SPAState,
    util::apply_lb_and_temp_to_spa,
    InfOutOptions, InferenceOutput, SPATree, SPA,
};
use anyhow::Result;
use bitvec::{prelude::Lsb0, vec::BitVec};
use bytes::{Buf, BufMut, Bytes};
use hashbrown::{HashMap, HashSet};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct DirichletSPATree {
    pub ns: Vec<u64>,
    pub branches: LZWTree,
    pub ghost_ns: Vec<u64>,
    alphabet_size: u32,
    pub num_nodes: u64,
}

impl ToFromBytes for DirichletSPATree {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.ns.to_bytes()?;
        bytes.extend(self.branches.to_bytes()?);
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.is_real_node.clone().into_vec().to_bytes()?);
        bytes.put_u64_le(self.num_nodes);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let ns = Vec::<u64>::from_bytes(bytes)?;
        let branches = LZWTree::from_bytes(bytes)?;
        let alphabet_size = bytes.get_u32_le();
        let is_real_node = BitVec::from_vec(Vec::<u64>::from_bytes(bytes)?);
        let num_nodes = bytes.get_u64_le();
        Ok(Self {
            ns,
            branches,
            is_real_node,
            alphabet_size,
            num_nodes,
        })
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
            Some(i) => self.ns[*i as usize],
            None => 0,
        } as f64;
        Ok((count + dirichlet_config.gamma)
            / (self.ns[idx as usize] as f64 - 1.0
                + dirichlet_config.gamma * dirichlet_config.alphabet_size as f64))
    }

    fn add_new_internal(&mut self, _config: &SPAConfig, parent_idx: u64, sym: u32) -> Result<()> {
        self.branches
            .add_leaf(parent_idx, sym, self.ns.len() as u64);
        self.ns.push(0);
        self.is_real_node.push(false);

        Ok(())
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

        if idx == 0 {
            self.ns[idx as usize] += 1;
        }
        let child_idx = match self.branches.get_child_idx(idx, sym) {
            Some(i) => *i as usize,
            None => {
                self.add_new_internal(config, idx, sym)?;
                self.ns.len() - 1
            }
        };
        self.ns[child_idx] += 1;
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
            if let Some(i) = self.branches.get_child_idx(idx, sym) {
                spa[sym as usize] = self.ns[*i as usize] as f64;
            }
        }

        spa = (spa + config.gamma)
            / (self.ns[idx as usize] as f64 - 1.0 + config.gamma * config.alphabet_size as f64);
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

    fn add_new(&mut self, config: &SPAConfig, parent_idx: u64, sym: u32) -> Result<()> {
        self.num_nodes += 1;
        match self.branches.get_child_idx(parent_idx, sym) {
            Some(i) => self.is_real_node.set(*i as usize, true),
            None => {
                self.add_new_internal(config, parent_idx, sym)?;
                self.is_real_node.set(self.ns.len() - 1, true);
            }
        }

        Ok(())
    }

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        match self.branches.get_child_idx(idx, sym) {
            Some(i) => {
                if self.is_real_node[*i as usize] {
                    Some(i)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    fn num_symbols_seen(&self, idx: u64) -> u64 {
        self.ns[idx as usize] - 1
    }

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let mut is_real_node = BitVec::new();
        is_real_node.push(true);
        Ok(Self {
            ns: vec![1],
            branches: LZWTree::new(),
            is_real_node,
            alphabet_size: config.alphabet_size(),
            num_nodes: 1,
        })
    }

    fn prune(&mut self, min_count: u64) {
        let mut remove = HashSet::new();
        let mut replace = HashMap::new();
        let mut write_idx = 0;
        for i in 0..self.ns.len() {
            if remove.contains(&(i as u64)) {
                if self.is_real_node[i] {
                    for a in 0..self.alphabet_size {
                        if let Some(child) = self.branches.get_child_idx(i as u64, a) {
                            remove.insert(*child);
                        }
                    }
                }
                continue;
            }
            replace.insert(i as u64, write_idx as u64);
            self.ns[write_idx] = self.ns[i];
            let is_real_node = self.is_real_node[i];
            self.is_real_node.set(write_idx, is_real_node);

            if self.ns[i] < min_count {
                self.is_real_node.set(write_idx, false);
                for a in 0..self.alphabet_size {
                    if let Some(child) = self.branches.get_child_idx(i as u64, a) {
                        remove.insert(*child);
                    }
                }
            }
            write_idx += 1;
        }
        self.ns.truncate(write_idx.max(1));
        self.is_real_node.truncate(write_idx.max(1));
        self.branches.remove_batch(&remove);
        self.branches.replace(&replace);
    }

    fn shrink_to_fit(&mut self) {
        self.branches.shrink_to_fit();
        self.ns.shrink_to_fit();
    }

    fn num_nodes(&self) -> u64 {
        self.num_nodes
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
