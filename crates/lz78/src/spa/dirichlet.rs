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
use bytes::{Buf, BufMut, Bytes};
use hashbrown::{HashMap, HashSet};
use ndarray::{Array1, ArrayViewMut1};

#[derive(Debug, Clone)]
pub struct DirichletSPATree {
    pub ns: Vec<u64>,
    pub branches: LZWTree,
    pub ghost_ns: HashMap<u64, u64>,
    alphabet_size: u32,
    pub next_ghost_node: u64,
}

impl ToFromBytes for DirichletSPATree {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.ns.to_bytes()?;
        bytes.extend(self.branches.to_bytes()?);
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.ghost_ns.to_bytes()?);
        bytes.put_u64_le(self.next_ghost_node);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let ns = Vec::<u64>::from_bytes(bytes)?;
        let branches = LZWTree::from_bytes(bytes)?;
        let alphabet_size = bytes.get_u32_le();
        let ghost_ns = HashMap::<u64, u64>::from_bytes(bytes)?;
        let next_ghost_node = bytes.get_u64_le();
        Ok(Self {
            ns,
            branches,
            ghost_ns,
            alphabet_size,
            next_ghost_node,
        })
    }
}

impl DirichletSPATree {
    fn spa_for_symbol_basic(
        &self,
        idx: u64,
        sym: u32,
        dirichlet_config: &DirichletConfig,
    ) -> Result<f32> {
        let count = match self.branches.get_child_idx(idx, sym) {
            Some(i) => self.get_count(*i),
            None => 0,
        } as f32;
        Ok((count + dirichlet_config.gamma as f32)
            / (self.get_count(idx) as f32 - 1.0
                + dirichlet_config.gamma as f32 * dirichlet_config.alphabet_size as f32))
    }

    fn add_new_ghost(&mut self, parent_idx: u64, sym: u32) -> Result<()> {
        let idx = (self.next_ghost_node as u64) | (1 << 63);
        self.next_ghost_node += 1;
        self.branches.add_leaf(parent_idx, sym, idx);
        self.ghost_ns.insert(idx, 0);

        Ok(())
    }

    fn get_count(&self, node_id: u64) -> u64 {
        if node_id & (1 << 63) != 0 {
            self.ghost_ns[&node_id]
        } else {
            self.ns[node_id as usize]
        }
    }

    fn increment(&mut self, node_id: u64) {
        if node_id & (1 << 63) != 0 {
            self.ghost_ns.insert(node_id, self.ghost_ns[&node_id] + 1);
        } else {
            self.ns[node_id as usize] += 1;
        }
    }
}

impl SPATree for DirichletSPATree {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        _state: &mut SPAState,
    ) -> Result<f32> {
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
            Some(i) => *i,
            None => {
                self.add_new_ghost(idx, sym)?;
                (self.next_ghost_node - 1) | (1 << 63)
            }
        };
        self.increment(child_idx);
        Ok(loss)
    }

    fn spa_for_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f32> {
        let dirichlet_config = config.try_get_dirichlet_mut()?;
        if dirichlet_config.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(idx, config, state, context_syms)?[sym as usize]);
        }
        self.spa_for_symbol_basic(idx, sym, &dirichlet_config)
    }

    fn spa_in_place(
        &self,
        idx: u64,
        config: &mut SPAConfig,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
        mut spa: ArrayViewMut1<f32>,
    ) -> Result<()> {
        let config = config.try_get_dirichlet_mut()?;

        for sym in 0..config.alphabet_size {
            if let Some(i) = self.branches.get_child_idx(idx, sym) {
                spa[sym as usize] = self.get_count(*i) as f32;
            }
        }
        let n = spa.sum();

        spa += config.gamma as f32;
        spa /= n + config.gamma as f32 * config.alphabet_size as f32;
        apply_lb_and_temp_to_spa(spa, config.lb_and_temp, None);
        Ok(())
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f32> {
        Ok(-self
            .spa_for_symbol(idx, sym, config, state, context_syms)?
            .log2())
    }

    fn add_new(&mut self, _config: &SPAConfig, parent_idx: u64, sym: u32) -> Result<()> {
        let idx = *self.branches.get_child_idx(parent_idx, sym).unwrap_or(&0);
        if idx == 0 {
            self.branches
                .add_leaf(parent_idx, sym, self.ns.len() as u64);
            self.ns.push(0);
            return Ok(());
        }
        self.branches.branches.remove(&(parent_idx, sym));
        self.branches
            .add_leaf(parent_idx, sym, self.ns.len() as u64);
        self.ns.push(self.ghost_ns[&idx]);
        self.ghost_ns.remove(&idx);

        Ok(())
    }

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        match self.branches.get_child_idx(idx, sym) {
            Some(i) => {
                if i & (1 << 63) == 0 {
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
        Ok(Self {
            ns: vec![1],
            branches: LZWTree::new(),
            ghost_ns: HashMap::new(),
            alphabet_size: config.alphabet_size(),
            next_ghost_node: 0,
        })
    }

    fn prune(&mut self, min_count: u64) {
        let mut remove = HashSet::new();
        let mut replace = HashMap::new();
        let mut write_idx = 0;
        for i in 0..self.ns.len() {
            if remove.contains(&(i as u64)) {
                for a in 0..self.alphabet_size {
                    if let Some(child) = self.branches.get_child_idx(i as u64, a) {
                        remove.insert(*child);
                    }
                }
                continue;
            }
            if self.ns[i] < min_count {
                for a in 0..self.alphabet_size {
                    if let Some(child) = self.branches.get_child_idx(i as u64, a) {
                        remove.insert(*child);
                    }
                }
                replace.insert(i as u64, self.next_ghost_node | (1 << 63));
                self.ghost_ns
                    .insert(self.next_ghost_node | (1 << 63), self.ns[i]);
                self.next_ghost_node += 1;
            } else {
                replace.insert(i as u64, write_idx as u64);
                self.ns[write_idx] = self.ns[i];

                write_idx += 1;
            }
        }
        for key in remove.iter() {
            self.ghost_ns.remove(key);
        }

        self.ns.truncate(write_idx.max(1));
        println!("Pruned to size {write_idx}");
        self.branches.remove_batch(&remove);
        self.branches.replace(&replace);
    }

    fn shrink_to_fit(&mut self) {
        self.branches.shrink_to_fit();
        self.ns.shrink_to_fit();
    }

    fn num_nodes(&self) -> u64 {
        self.ns.len() as u64
    }
}

impl GenerationSPATree for DirichletSPATree {
    fn input_seed_data_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f32> {
        self.test_on_symbol(idx, sym, config, state, None)
    }

    fn generate_one_symbol(
        &self,
        idx: u64,
        rng_sample: f32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f32,
        topk: Option<u32>,
    ) -> Result<(u32, f32)> {
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
    pub counts: Array1<f32>,
}

impl SPA for DirichletSPA {
    fn train_on_symbol(
        &mut self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f32> {
        let loss = -self.spa_for_symbol(sym, config, state, None)?.log2();
        self.counts[sym as usize] += 1.0;
        self.n += 1;
        Ok(loss)
    }

    fn spa_in_place(
        &self,
        config: &mut SPAConfig,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
        mut output: ArrayViewMut1<f32>,
    ) -> Result<()> {
        let config = config.try_get_dirichlet_mut()?;

        output.assign(&(&self.counts + config.gamma as f32));
        output /= self.n as f32 + config.gamma as f32 * config.alphabet_size as f32;
        apply_lb_and_temp_to_spa(output, config.lb_and_temp, None);
        Ok(())
    }

    fn spa_for_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f32> {
        let dirichlet_config = config.try_get_dirichlet_mut()?;
        if dirichlet_config.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(config, state, context_syms)?[sym as usize]);
        }
        Ok((self.counts[sym as usize] + dirichlet_config.gamma as f32)
            / (self.n as f32
                + dirichlet_config.gamma as f32 * dirichlet_config.alphabet_size as f32))
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
        prob_dist_output: Option<ArrayViewMut1<f32>>,
    ) -> Result<InferenceOutput> {
        if inf_out_options.output_probs() {
            if let Some(mut dist) = prob_dist_output {
                self.spa_in_place(config, state, context_syms, dist.view_mut())?;
                let loss = -dist[sym as usize].log2();
                let ppl = loss.exp2();
                return Ok(InferenceOutput::new(
                    loss,
                    ppl,
                    vec![loss],
                    vec![dist.to_vec()],
                ));
            } else {
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
    ) -> Result<f32> {
        Ok(self
            .test_on_symbol(sym, config, state, InfOutOptions::Basic, None, None)?
            .avg_log_loss)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f32,
        topk: Option<u32>,
    ) -> Result<(u32, f32)> {
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
