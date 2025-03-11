use crate::{spa::util::LbAndTemp, storage::ToFromBytes};

use super::{
    generation::{gen_symbol_from_spa, GenerationSPA, GenerationSPATree},
    params::{DirichletParams, SPAParams},
    states::SPAState,
    util::apply_lb_and_temp_to_spa,
    LZWTree, SPATree, SPA,
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
        dir_params: &DirichletParams,
    ) -> Result<f64> {
        let count = match self.branches.get_child_idx(idx, sym) {
            Some(i) => self.ns[*i as usize] + 1,
            None => 0,
        } as f64;
        Ok((count + dir_params.gamma)
            / (self.ns[idx as usize] as f64 + dir_params.gamma * dir_params.alphabet_size as f64))
    }
}

impl SPATree for DirichletSPATree {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        _state: &mut SPAState,
    ) -> Result<f64> {
        let loss = if params.compute_training_loss() {
            -self
                .spa_for_symbol_basic(idx, sym, params.try_get_dirichlet_mut()?)?
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
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let dir_params = params.try_get_dirichlet_mut()?;
        if dir_params.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(idx, params, state, context_syms)?[sym as usize]);
        }
        self.spa_for_symbol_basic(idx, sym, &dir_params)
    }

    fn spa(
        &self,
        idx: u64,
        params: &mut SPAParams,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let params = params.try_get_dirichlet_mut()?;

        let mut spa = Array1::zeros(params.alphabet_size as usize);
        for sym in 0..params.alphabet_size {
            if let Some(i) = self.get_child_idx(idx, sym) {
                spa[sym as usize] = self.ns[*i as usize] as f64 + 1.0;
            }
        }

        spa = (spa + params.gamma)
            / (self.ns[idx as usize] as f64 + params.gamma * params.alphabet_size as f64);
        apply_lb_and_temp_to_spa(&mut spa, params.lb_and_temp, None);
        Ok(spa)
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-self
            .spa_for_symbol(idx, sym, params, state, context_syms)?
            .log2())
    }

    fn add_new(&mut self, _params: &SPAParams, parent_idx: u64, sym: u32) -> Result<()> {
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

    fn new(_params: &SPAParams) -> Result<Self>
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
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.test_on_symbol(idx, sym, params, state, None)
    }

    fn generate_one_symbol(
        &self,
        idx: u64,
        rng_sample: f64,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(
            rng_sample,
            &self.spa(idx, params, state, Some(context_syms))?,
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
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let loss = -self.spa_for_symbol(sym, params, state, None)?.log2();
        self.counts[sym as usize] += 1.0;
        self.n += 1;
        Ok(loss)
    }

    fn spa(
        &self,
        params: &mut SPAParams,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let params = params.try_get_dirichlet_mut()?;
        let mut spa = self.counts.clone() * params.gamma
            / (self.n as f64 + params.gamma * params.alphabet_size as f64);
        apply_lb_and_temp_to_spa(&mut spa, params.lb_and_temp, None);
        Ok(spa)
    }

    fn spa_for_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let dir_params = params.try_get_dirichlet_mut()?;
        if dir_params.lb_and_temp != LbAndTemp::Skip {
            return Ok(self.spa(params, state, context_syms)?[sym as usize]);
        }
        Ok((self.counts[sym as usize] + dir_params.gamma)
            / (self.n as f64 + dir_params.gamma * dir_params.alphabet_size as f64))
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-self
            .spa_for_symbol(sym, params, state, context_syms)?
            .log2())
    }

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            n: 0,
            counts: Array1::zeros(params.alphabet_size() as usize),
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
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.test_on_symbol(sym, params, state, None)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(
            rng_sample,
            &self.spa(params, state, Some(context_syms))?,
            temperature,
            topk,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::spa::params::DirichletParamsBuilder;

    use super::*;

    #[test]
    fn test_dirichlet_tree_to_from_bytes() {
        let mut params = DirichletParamsBuilder::new(3)
            .gamma(0.2)
            .lb_and_temp(0.1, 2.0, true)
            .build_enum();
        let mut state = SPAState::None;
        let mut spa = DirichletSPATree::new(&params).expect("failed to make DirichletSPA");
        spa.train_spa_on_symbol(0, 0, &mut params, &mut state)
            .expect("train dirichlet spa failed");
        spa.train_spa_on_symbol(0, 2, &mut params, &mut state)
            .expect("train dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPATree::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.ns, new_spa.ns);
    }
}
