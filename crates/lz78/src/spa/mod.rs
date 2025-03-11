use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};
use hashbrown::{HashMap, HashSet};
use ndarray::Array1;
use params::SPAParams;
use states::SPAState;

use crate::{sequence::Sequence, storage::ToFromBytes};

pub mod dirichlet;
pub mod generation;
pub mod lz_transform;
pub mod params;
pub mod states;
pub mod util;

#[derive(Debug, Clone)]
pub struct LZWTree {
    pub branches: HashMap<(u64, u32), u64>,
}

unsafe impl Sync for LZWTree {}

impl ToFromBytes for LZWTree {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.branches.len() as u64);
        for ((k1, k2), v) in self.branches.iter() {
            bytes.put_u64_le(*k1);
            bytes.put_u32_le(*k2);
            bytes.put_u64_le(*v);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le() as usize;
        let mut branches = HashMap::with_capacity(n);

        for _ in 0..n {
            let (k1, k2, v) = (bytes.get_u64_le(), bytes.get_u32_le(), bytes.get_u64_le());
            branches.insert((k1, k2), v);
        }

        Ok(Self { branches })
    }
}

impl LZWTree {
    pub fn new() -> Self {
        Self {
            branches: HashMap::new(),
        }
    }

    pub fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        self.branches.get(&(idx, sym))
    }

    pub fn add_leaf(&mut self, idx: u64, sym: u32, child_idx: u64) {
        self.branches.insert((idx, sym), child_idx);
    }

    pub fn remove_batch(&mut self, nodes: &HashSet<u64>) {
        self.branches = self
            .branches
            .iter()
            .filter(|((parent, _), child)| !nodes.contains(parent) && !nodes.contains(*child))
            .map(|((parent, sym), child)| ((*parent, *sym), *child))
            .collect();
    }

    pub fn replace(&mut self, node_map: &HashMap<u64, u64>) {
        self.branches = self
            .branches
            .iter()
            .map(|((parent, sym), child)| {
                (
                    (*node_map.get(parent).unwrap_or(parent), *sym),
                    *node_map.get(child).unwrap_or(child),
                )
            })
            .collect();
    }

    pub fn shrink_to_fit(&mut self) {
        self.branches.shrink_to_fit();
    }
}

pub trait SPATree: Sync {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64>;

    fn spa_for_symbol(
        &self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn spa(
        &self,
        idx: u64,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let mut spa = Array1::zeros(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa[sym as usize] = self.spa_for_symbol(idx, sym, params, state, context_syms)?;
        }
        Ok(spa)
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn add_new(&mut self, params: &SPAParams, parent_idx: u64, sym: u32) -> Result<()>;

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64>;

    fn num_symbols_seen(&self, idx: u64) -> u64;

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized;

    fn prune(&mut self, min_count: u64);

    fn shrink_to_fit(&mut self);
}

pub trait SPA {
    fn train_on_block<T: ?Sized>(
        &mut self,
        input: &T,
        params: &mut SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sym in input.iter() {
            loss += self.train_on_symbol(sym, params, train_state)?
        }
        Ok(loss)
    }

    fn train_on_symbol(
        &mut self,
        sym: u32,
        params: &mut SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64>;

    fn spa_for_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn spa(
        &self,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let mut spa = Array1::zeros(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa[sym as usize] = self.spa_for_symbol(sym, params, state, context_syms)?;
        }
        Ok(spa)
    }

    fn test_on_block<T: ?Sized>(
        &self,
        input: &T,
        params: &mut SPAParams,
        inference_state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss: f64 = 0.;
        let mut syms = if let Some(syms) = context_syms {
            syms.to_vec()
        } else {
            Vec::new()
        };
        syms.reserve(input.len() as usize);
        for sym in input.iter() {
            loss += self.test_on_symbol(sym, params, inference_state, Some(&syms))?;
            syms.push(sym);
        }
        Ok(loss)
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        inference_state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized;

    fn num_symbols_seen(&self) -> u64;
}
