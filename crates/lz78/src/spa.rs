use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, bail, Result};
use bitvec::vec::BitVec;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

use crate::{
    generation::{GenerationParams, GenerationSPA},
    sequence::Sequence,
    storage::ToFromBytes,
    util::sample_from_pdf,
};

#[derive(Debug, Clone, Copy)]
pub struct DirichletSPAParams {
    alphabet_size: u32,
    gamma: f64,
}

#[derive(Debug, Clone)]
pub struct LZ78SPAParams {
    alphabet_size: u32,
    inner_params: Arc<SPAParams>,
    debug: bool,
}

#[derive(Debug, Clone)]
pub enum SPAParams {
    Dirichlet(DirichletSPAParams),
    LZ78(LZ78SPAParams),
}

impl SPAParams {
    pub fn new_dirichlet(alphabet_size: u32, gamma: f64) -> Self {
        Self::Dirichlet(DirichletSPAParams {
            alphabet_size,
            gamma,
        })
    }

    pub fn new_lz78(inner_spa_params: SPAParams, debug: bool) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size: inner_spa_params.alphabet_size(),
            inner_params: Arc::new(inner_spa_params),
            debug,
        })
    }

    pub fn new_lz78_dirichlet(alphabet_size: u32, gamma: f64, debug: bool) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Arc::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma,
            })),
            debug,
        })
    }

    pub fn default_lz78_dirichlet(alphabet_size: u32) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Arc::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma: 0.5,
            })),
            debug: false,
        })
    }

    pub fn alphabet_size(&self) -> u32 {
        match self {
            SPAParams::Dirichlet(params) => params.alphabet_size,
            SPAParams::LZ78(params) => params.alphabet_size,
        }
    }
}

impl ToFromBytes for SPAParams {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SPAParams::Dirichlet(dirichlet_spaparams) => {
                bytes.put_u8(0);
                bytes.put_u32_le(dirichlet_spaparams.alphabet_size);
                bytes.put_f64_le(dirichlet_spaparams.gamma);
            }
            SPAParams::LZ78(lz78_spaparams) => {
                bytes.put_u8(1);
                bytes.put_u32_le(lz78_spaparams.alphabet_size);
                bytes.put_u8(lz78_spaparams.debug as u8);
                bytes.extend(lz78_spaparams.inner_params.to_bytes()?);
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let tpe = bytes.get_u8();
        match tpe {
            0 => {
                let alphabet_size = bytes.get_u32_le();
                let gamma = bytes.get_f64_le();
                Ok({
                    Self::Dirichlet(DirichletSPAParams {
                        alphabet_size,
                        gamma,
                    })
                })
            }
            1 => {
                let alphabet_size = bytes.get_u32_le();
                let debug = bytes.get_u8() == 1;
                let inner_params = Self::from_bytes(bytes)?;
                Ok(Self::LZ78(LZ78SPAParams {
                    alphabet_size,
                    inner_params: Arc::new(inner_params),
                    debug,
                }))
            }
            _ => bail!("Unexpected SPA type indicator {tpe}"),
        }
    }
}

pub trait SPA {
    fn train_on_block<T: ?Sized>(&mut self, input: &T, params: &SPAParams) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sym in input.iter() {
            loss += self.train_on_symbol(sym, params)?
        }
        Ok(loss)
    }

    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64>;

    fn spa_for_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64>;

    fn spa(&mut self, params: &SPAParams) -> Result<Vec<f64>> {
        let mut spa = Vec::with_capacity(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa.push(self.spa_for_symbol(sym, params)?);
        }
        Ok(spa)
    }

    fn test_on_block<T: ?Sized>(&mut self, input: &T, params: &SPAParams) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss: f64 = 0.;
        for sym in input.iter() {
            loss += self.test_on_symbol(sym, params)?;
        }
        Ok(loss)
    }

    fn test_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64>;

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized;

    fn reset_state(&mut self);

    fn num_symbols_seen(&self) -> u64;
}

pub struct DirichletSPA {
    counts: HashMap<u32, u64>,
    n: u64,
}

impl SPA for DirichletSPA {
    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let loss = -self.spa_for_symbol(input, params)?.log2();
        self.counts
            .insert(input, self.counts.get(&input).unwrap_or(&0) + 1);
        self.n += 1;
        Ok(loss)
    }

    fn spa_for_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        if let SPAParams::Dirichlet(params) = params {
            let sym_count = *self.counts.get(&sym).unwrap_or(&0) as f64;
            Ok((sym_count + params.gamma)
                / (self.n as f64 + params.gamma * params.alphabet_size as f64))
        } else {
            bail!("Wrong SPA parameters passed in for Dirichlet SPA");
        }
    }

    fn test_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        Ok(-self.spa_for_symbol(input, params)?.log2())
    }

    fn new(_params: &SPAParams) -> Result<Self> {
        Ok(Self {
            counts: HashMap::new(),
            n: 0,
        })
    }

    /// There is no state to reset
    fn reset_state(&mut self) {}

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl ToFromBytes for DirichletSPA {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.counts.len() as u32);
        for (&sym, &count) in self.counts.iter() {
            bytes.put_u32_le(sym);
            bytes.put_u64_le(count);
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let counts_len = bytes.get_u32_le();

        let mut counts: HashMap<u32, u64> = HashMap::new();
        let mut n = 0;
        for _ in 0..counts_len {
            let sym = bytes.get_u32_le();
            let count = bytes.get_u64_le();
            n += count;
            counts.insert(sym, count);
        }

        Ok(Self { counts, n })
    }
}

//. All nodes in the LZ78 tree are stored in the `spas` array, and the tree
/// structure is encoded via the `branch_mappings` array. There is a one-to-
/// one correspondence between the elements of `spas` and `branch_mappings`,
/// i.e., each node of the LZ78 tree includes a SPA and map of its child nodes.
/// For instance, consider the sequence
///     00010111,
/// which is parsed into phrases as 0, 00, 1, 01, 11, would have the tree
/// structure:
///```ignore
///                                []
///                           [0]      [1]
///                       [00]  [01]      [11],
/// ```
///
/// and the nodes would be stored in the root of the tree in the same order as
/// the parsed phrases. The root always has index 0, so, in this example, "0"
/// would have index 1, "00" would have index 2, etc.. In that case, the first
/// element of `branch_mappings` (corresponding to the root) would be
/// `{0 -> 1, 1 -> 3}`, the node "0" would have branches `{0 -> 2, 1 -> 4}`,
/// and the node "1" would have branches `{1 -> 5}`.
struct SPATree<S> {
    spas: Vec<S>,
    branch_mappings: Vec<HashMap<u32, u64>>,
    pending_reset: BitVec<u64>,
    params: Arc<SPAParams>,
}

impl<S> SPATree<S> {
    const ROOT_IDX: u64 = 0;

    fn new(params: Arc<SPAParams>) -> Result<Self>
    where
        S: SPA,
    {
        let mut pending_reset = BitVec::new();
        pending_reset.push(false);
        Ok(Self {
            spas: vec![S::new(&params)?],
            branch_mappings: vec![HashMap::new()],
            pending_reset,
            params,
        })
    }

    fn traverse_one_symbol_frozen(&self, state: u64, sym: u32) -> u64 {
        if self.branch_mappings[state as usize].contains_key(&sym) {
            self.branch_mappings[state as usize][&sym]
        } else {
            Self::ROOT_IDX
        }
    }

    fn traverse_one_symbol_and_maybe_grow(&mut self, state: u64, sym: u32) -> Result<u64>
    where
        S: SPA,
    {
        let new_state = self.traverse_one_symbol_frozen(state, sym);
        if new_state == Self::ROOT_IDX {
            // add a new leaf
            let new_node_idx = self.spas.len() as u64;
            let new_spa = S::new(&self.params)?;

            self.spas.push(new_spa);
            self.branch_mappings[state as usize].insert(sym, new_node_idx);
            self.branch_mappings.push(HashMap::new());
            self.pending_reset.push(false);
        }

        Ok(new_state)
    }

    fn train_on_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].train_on_symbol(sym, &self.params)
    }

    fn test_on_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].test_on_symbol(sym, &self.params)
    }

    fn spa_for_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].spa_for_symbol(sym, &self.params)
    }

    fn spa(&mut self, state: u64) -> Result<Vec<f64>>
    where
        S: SPA,
    {
        let mut result: Vec<f64> = Vec::with_capacity(self.params.alphabet_size() as usize);
        for sym in 0..self.params.alphabet_size() {
            result.push(self.spa_for_symbol(state, sym)?);
        }

        Ok(result)
    }

    fn recursively_reset_state(&mut self) {
        self.pending_reset.fill(true);
    }
}

impl<S> ToFromBytes for SPATree<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.spas.len() as u64);
        for (spa, branches) in self.spas.iter().zip(self.branch_mappings.iter()) {
            bytes.extend(spa.to_bytes()?);

            // branch_mappings
            bytes.put_u32_le(branches.len() as u32);
            for (&sym, &child) in branches.iter() {
                bytes.put_u32_le(sym);
                bytes.put_u64_le(child);
            }
        }

        let pending_reset_bytes = self.pending_reset.as_raw_slice();
        bytes.put_u64_le(pending_reset_bytes.len() as u64);
        for &x in pending_reset_bytes {
            bytes.put_u64_le(x);
        }
        bytes.extend(self.params.to_bytes()?);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let n_spas = bytes.get_u64_le();
        let mut spas: Vec<S> = Vec::with_capacity(n_spas as usize);
        let mut branch_mappings: Vec<HashMap<u32, u64>> = Vec::with_capacity(n_spas as usize);
        for _ in 0..n_spas {
            spas.push(S::from_bytes(bytes)?);

            let n_branches = bytes.get_u32_le();
            let mut branches: HashMap<u32, u64> = HashMap::with_capacity(n_branches as usize);
            for _ in 0..n_branches {
                let sym = bytes.get_u32_le();
                let child = bytes.get_u64_le();
                branches.insert(sym, child);
            }
            branch_mappings.push(branches);
        }

        let pending_reset_len = bytes.get_u64_le();
        let mut pending_reset: Vec<u64> = Vec::with_capacity(pending_reset_len as usize);
        for _ in 0..pending_reset_len {
            pending_reset.push(bytes.get_u64_le());
        }
        let pending_reset = BitVec::from_vec(pending_reset);
        let params = SPAParams::from_bytes(bytes)?;

        Ok(Self {
            spas,
            branch_mappings,
            pending_reset,
            params: Arc::new(params),
        })
    }
}

struct LZ78GenerationState {
    tree_state: u64,
    nodes_seen: HashSet<u64>,
    seq: Vec<u32>,
    last_time_root_seen: u64,
}

impl LZ78GenerationState {
    fn new() -> Self {
        Self {
            tree_state: 0,
            nodes_seen: HashSet::new(),
            seq: Vec::new(),
            last_time_root_seen: 0,
        }
    }

    fn clear(&mut self) {
        self.tree_state = 0;
        self.nodes_seen.clear();
        self.seq.clear();
        self.last_time_root_seen = 0;
    }
}

/// Tracks, e.g., the depth of each leaf for better understanding the tree
/// built by the LZ78 SPA
#[derive(Debug, Clone)]
pub struct LZ78DebugState {
    pub max_depth: u32,
    pub current_depth: u32,
    pub deepest_leaf: u64,
    pub leaf_depths: HashMap<u64, u32>,
    pub parent_and_sym_map: HashMap<u64, (u64, u32)>,
}

impl LZ78DebugState {
    fn new() -> Self {
        Self {
            max_depth: 0,
            current_depth: 0,
            deepest_leaf: 0,
            leaf_depths: HashMap::new(),
            parent_and_sym_map: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.leaf_depths.clear();
        self.parent_and_sym_map.clear();
    }

    fn add_leaf(&mut self, state: u64, new_leaf_idx: u64, new_leaf_sym: u32, leaf_depth: u32) {
        if leaf_depth > self.max_depth {
            self.max_depth = leaf_depth;
            self.deepest_leaf = new_leaf_idx;
        }
        if self.leaf_depths.contains_key(&state) {
            self.leaf_depths.remove(&state);
        }
        self.leaf_depths.insert(new_leaf_idx, leaf_depth);
        self.parent_and_sym_map
            .insert(new_leaf_idx, (state, new_leaf_sym));
    }

    pub fn get_longest_branch(&self) -> Vec<u32> {
        let mut res: Vec<u32> = Vec::new();

        let mut node = self.deepest_leaf;
        while node != 0 {
            let (parent, sym) = self.parent_and_sym_map[&node];
            res.push(sym);
            node = parent;
        }

        res.into_iter().rev().collect_vec()
    }
}

impl ToFromBytes for LZ78DebugState {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.max_depth);
        bytes.put_u32_le(self.current_depth);
        bytes.put_u64_le(self.deepest_leaf);
        bytes.put_u64_le(self.leaf_depths.len() as u64);
        for (&k, &v) in self.leaf_depths.iter() {
            bytes.put_u64_le(k);
            bytes.put_u32_le(v);
        }

        bytes.put_u64_le(self.parent_and_sym_map.len() as u64);
        for (&k, &v) in self.parent_and_sym_map.iter() {
            bytes.put_u64_le(k);
            bytes.put_u64_le(v.0);
            bytes.put_u32_le(v.1);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let max_depth = bytes.get_u32_le();
        let current_depth = bytes.get_u32_le();
        let deepest_leaf = bytes.get_u64_le();

        let n_leaves = bytes.get_u64_le();
        let mut leaf_depths: HashMap<u64, u32> = HashMap::new();
        for _ in 0..n_leaves {
            let k = bytes.get_u64_le();
            let v = bytes.get_u32_le();
            leaf_depths.insert(k, v);
        }

        let n_nodes = bytes.get_u64_le();
        let mut parent_and_sym_map: HashMap<u64, (u64, u32)> = HashMap::new();
        for _ in 0..n_nodes {
            let k = bytes.get_u64_le();
            let v = (bytes.get_u64_le(), bytes.get_u32_le());
            parent_and_sym_map.insert(k, v);
        }

        Ok(Self {
            max_depth,
            current_depth,
            leaf_depths,
            parent_and_sym_map,
            deepest_leaf,
        })
    }
}

pub fn lz78_spa_monte_carlo_branch_lengths<S>(
    spa: &mut LZ78SPA<S>,
    n_trials: u32,
) -> Result<Vec<u32>>
where
    S: SPA,
{
    let mut state = SPATree::<S>::ROOT_IDX;
    let mut results: Vec<u32> = Vec::new();

    let mut rng = thread_rng();
    let mut rand_iter = Uniform::new(0.0, 1.0).sample_iter(&mut rng);

    for _ in 0..n_trials {
        let mut depth: u32 = 0;
        while depth == 0 || state != SPATree::<S>::ROOT_IDX {
            depth += 1;
            let pdf = spa.spa_tree.spa(state)?;
            let sym = sample_from_pdf(
                &pdf,
                rand_iter
                    .next()
                    .ok_or_else(|| anyhow!("could not get sample from rng"))?,
            ) as u32;
            state = spa.spa_tree.traverse_one_symbol_frozen(state, sym);
        }
        results.push(depth - 1);
    }

    Ok(results)
}
/// LZ78 implementation of the sequential probability assignment
pub struct LZ78SPA<S> {
    spa_tree: SPATree<S>,
    state: u64,
    n: u64,
    total_log_loss: f64,
    gen_state: LZ78GenerationState,
    debug: LZ78DebugState,
}

impl<S> LZ78SPA<S> {
    pub fn get_normalized_log_loss(&self) -> f64 {
        self.total_log_loss / self.n as f64
    }

    pub fn get_debug_info(&self) -> &LZ78DebugState {
        &self.debug
    }

    pub fn clear_debug_info(&mut self) {
        self.debug.clear();
    }

    pub fn get_n(&self) -> u64 {
        self.n
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPA,
{
    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let params = if let SPAParams::LZ78(p) = params {
            p
        } else {
            bail!("Invalid SPAParams for LZ78 SPA")
        };

        let loss = self.spa_tree.train_on_symbol(self.state, input)?;
        self.total_log_loss += loss;

        let old_state = self.state;

        self.state = self
            .spa_tree
            .traverse_one_symbol_and_maybe_grow(self.state, input)?;

        if params.debug {
            if self.state == SPATree::<S>::ROOT_IDX {
                self.debug.add_leaf(
                    old_state,
                    self.spa_tree.spas.len() as u64 - 1,
                    input,
                    self.debug.current_depth,
                );
                self.debug.current_depth = 0;
            } else {
                self.debug.current_depth += 1;
            }
        }
        self.n += 1;

        Ok(loss)
    }

    fn spa_for_symbol(&mut self, sym: u32, _params: &SPAParams) -> Result<f64> {
        self.spa_tree.spa_for_symbol(self.state, sym)
    }

    fn test_on_symbol(&mut self, input: u32, _params: &SPAParams) -> Result<f64> {
        let loss = self.spa_tree.test_on_symbol(self.state, input)?;
        self.state = self.spa_tree.traverse_one_symbol_frozen(self.state, input);

        Ok(loss)
    }

    fn new(params: &SPAParams) -> Result<Self> {
        let params = if let SPAParams::LZ78(x) = params {
            x.inner_params.clone()
        } else {
            bail!("Wrong params for building LZ78 SPA")
        };
        Ok(Self {
            spa_tree: SPATree::new(params)?,
            state: SPATree::<S>::ROOT_IDX,
            n: 0,
            total_log_loss: 0.0,
            debug: LZ78DebugState::new(),
            gen_state: LZ78GenerationState::new(),
        })
    }

    fn reset_state(&mut self) {
        self.state = SPATree::<S>::ROOT_IDX;
        self.debug.current_depth = 0;
        self.spa_tree.recursively_reset_state();
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl<S> LZ78SPA<S>
where
    S: SPA,
{
    /// Traverse the tree from the root with given input sequence and return
    /// the probability of the next symbol for all symbols in the alphabet
    pub fn traverse_and_get_prob(&mut self, input: &[u32]) -> Result<Vec<f64>> {
        let mut state = SPATree::<S>::ROOT_IDX;
        for &sym in input {
            state = self.spa_tree.traverse_one_symbol_frozen(state, sym);
        }
        let pdf = self.spa_tree.spa(state)?;
        Ok(pdf)
    }

    /// Traverse the tree from the root with given input sequence and for all
    /// symbols in the alphabet, continue traversing the tree with the
    /// lookahead symbols and return the num_symbols_seen of the last symbol in
    /// the sequence.
    ///
    /// Normalize the probabilities of the num_symbols_seen of the last symbol
    /// in the sequence over all symbols in the alphabet, and return the
    /// normalized probabilities
    pub fn traverse_and_get_prob_with_lookahead(
        &mut self,
        input: &[u32],
        lookahead: &[u32],
    ) -> Result<Vec<f64>> {
        let alpha_size = self.spa_tree.params.alphabet_size();
        // iterate over all symbols in the alphabet
        // get the num_symbols_seen of the last symbol in the sequence

        let mut pdfs: Vec<f64> = Vec::new();
        for sym in 0..alpha_size {
            // get the num_symbols_seen of the last symbol in the sequence
            let mut state = SPATree::<S>::ROOT_IDX;
            // seed_data being input + sym+ lookahead
            let mut seed_data: Vec<u32> = input.to_vec();
            seed_data.push(sym);
            seed_data.extend(lookahead.iter());
            //println!("seed_data: {:?}", seed_data);
            // set flag to True
            let mut flag = true;
            for sym in seed_data {
                state = self.spa_tree.traverse_one_symbol_frozen(state, sym);
                // check if the state is root, if so break and pdfs push 0.0
                if state == SPATree::<S>::ROOT_IDX {
                    pdfs.push(0.0);
                    flag = false;
                    break;
                }
            }
            // get the num_symbols_seen of the last symbol in the sequence
            if flag {
                pdfs.push(self.spa_tree.spas[state as usize].num_symbols_seen() as f64);
            }
        }
        //print!("pdfs: {:?}", pdfs);
        pdfs = pdfs
            .iter()
            .map(|x| *x as f64 / pdfs.iter().sum::<f64>())
            .collect_vec();
        Ok(pdfs)
    }
}

impl<S> ToFromBytes for LZ78SPA<S>
where
    S: ToFromBytes,
{
    /// TODO: should we also store the generation state, etc.?
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes = self.spa_tree.to_bytes()?;
        bytes.put_u64_le(self.state);
        bytes.put_u64_le(self.n);
        bytes.put_f64_le(self.total_log_loss);
        bytes.extend(self.debug.to_bytes()?);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let spa_tree = SPATree::<S>::from_bytes(bytes)?;
        let state = bytes.get_u64_le();
        let n = bytes.get_u64_le();
        let total_log_loss = bytes.get_f64_le();
        let debug = LZ78DebugState::from_bytes(bytes)?;

        Ok(Self {
            spa_tree,
            state,
            n,
            total_log_loss,
            debug,
            gen_state: LZ78GenerationState::new(),
        })
    }
}

// Sequence Generation

impl GenerationSPA for DirichletSPA {
    fn cleanup_post_generation(&mut self) {}

    fn input_seed_data_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        self.test_on_symbol(sym, params)
    }

    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, f64)> {
        // Compute the probability, according to the LZ78 SPA, that the
        // next symbol is x, for every x in the alphabet
        let mut spa = self.spa(params)?;
        let most_likely_next_sym = (0..params.alphabet_size())
            .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
            .unwrap();

        // if temperature is 0.0, we just compute the argmax of the SPA. If
        // temperature is 1.0, the symbols are generated directly from the
        // SPA. In either case, we do not need the following computation.
        if gen_params.temperature != 0.0 && gen_params.temperature != 1.0 {
            spa = spa
                .iter()
                .map(|x| 2.0_f64.powf(x.log2() / gen_params.temperature))
                .collect_vec();
        }

        // top-k sampling
        (0..params.alphabet_size())
            .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
            .take((params.alphabet_size() - gen_params.top_k) as usize)
            .map(|i| {
                spa[i as usize] = 0.0;
            })
            .collect_vec();

        let sum: f64 = spa.iter().sum();
        spa = spa.iter().map(|x| *x / sum).collect_vec();

        let new_sym = if gen_params.temperature == 0.0 {
            most_likely_next_sym
        } else {
            sample_from_pdf(&spa, rng_sample) as u32
        };
        let loss = self.test_on_symbol(new_sym, params)?;
        Ok((new_sym, loss))
    }
}

impl<S> SPATree<S>
where
    S: GenerationSPA,
{
    fn input_seed_data_symbol(&mut self, state: u64, sym: u32, _params: &SPAParams) -> Result<f64> {
        self.spas[state as usize].input_seed_data_symbol(sym, &self.params)
    }

    fn generate_one_symbol(
        &mut self,
        state: u64,
        gen_params: &GenerationParams,
        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        self.spas[state as usize].generate_one_symbol(rng_sample, &self.params, gen_params)
    }
}

impl<S> GenerationSPA for LZ78SPA<S>
where
    S: GenerationSPA,
{
    fn cleanup_post_generation(&mut self) {
        for &spa_idx in self.gen_state.nodes_seen.iter() {
            self.spa_tree.spas[spa_idx as usize].cleanup_post_generation();
        }
        self.gen_state.clear();
    }

    fn input_seed_data_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        if self.gen_state.tree_state == SPATree::<S>::ROOT_IDX {
            self.gen_state.last_time_root_seen = self.gen_state.seq.len() as u64;
        }

        self.gen_state.nodes_seen.insert(self.gen_state.tree_state);

        let loss = self
            .spa_tree
            .input_seed_data_symbol(self.gen_state.tree_state, sym, params)?;
        self.gen_state.seq.push(sym);

        self.gen_state.tree_state = self
            .spa_tree
            .traverse_one_symbol_frozen(self.gen_state.tree_state, sym);

        Ok(loss)
    }

    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        _params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, f64)> {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context
        if self.gen_state.tree_state == SPATree::<S>::ROOT_IDX
            || self.spa_tree.spas[self.gen_state.tree_state as usize].num_symbols_seen()
                < gen_params.min_spa_training_points
        {
            // keep on trying to re-seed the SPA
            for start_idx in (self.gen_state.last_time_root_seen + 1).max(
                self.gen_state.seq.len() as u64
                    - gen_params
                        .desired_context_length
                        .min(self.gen_state.seq.len() as u64),
            )..(self.gen_state.seq.len() as u64)
            {
                self.gen_state.last_time_root_seen = start_idx;
                self.gen_state.tree_state = SPATree::<S>::ROOT_IDX;
                for &sym in self.gen_state.seq.iter().skip(start_idx as usize) {
                    self.gen_state.tree_state = self
                        .spa_tree
                        .traverse_one_symbol_frozen(self.gen_state.tree_state, sym);
                    if self.gen_state.tree_state == SPATree::<S>::ROOT_IDX {
                        break;
                    }
                }

                // re-seeding was successful!
                if self.gen_state.tree_state != SPATree::<S>::ROOT_IDX
                    && self.spa_tree.spas[self.gen_state.tree_state as usize].num_symbols_seen()
                        >= gen_params.min_spa_training_points
                {
                    break;
                }
            }
        }
        // if reseeding failed, we don't want to end up at a leaf!
        if self.spa_tree.spas[self.gen_state.tree_state as usize].num_symbols_seen() == 0 {
            self.gen_state.tree_state = SPATree::<S>::ROOT_IDX;
            self.gen_state.last_time_root_seen = self.gen_state.seq.len() as u64;
        }

        self.gen_state.nodes_seen.insert(self.gen_state.tree_state);
        let (new_sym, sym_loss) =
            self.spa_tree
                .generate_one_symbol(self.gen_state.tree_state, gen_params, rng_sample)?;
        self.gen_state.seq.push(new_sym);
        self.gen_state.tree_state = self
            .spa_tree
            .traverse_one_symbol_frozen(self.gen_state.tree_state, new_sym);
        Ok((new_sym, sym_loss))
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, U8Sequence};
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn test_dirichlet_to_from_bytes() {
        let params = SPAParams::new_dirichlet(2, 0.2);
        let mut spa = DirichletSPA::new(&params).expect("failed to make DirichletSPA");
        spa.train_on_block(
            &U8Sequence::from_data(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 2, 2, 2, 1], 3)
                .unwrap(),
            &params,
        )
        .expect("train dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPA::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.counts, new_spa.counts);
    }

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");

        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &params,
            )
            .expect("failed to compute test loss");
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                &params,
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_lz_transformed_nodes_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");

        let nodes_bytes = spa.spa_tree.to_bytes().expect("spa tree to bytes failed");
        let mut bytes: Bytes = nodes_bytes.into();
        let nodes =
            SPATree::<DirichletSPA>::from_bytes(&mut bytes).expect("spa tree from bytes failed");
        assert_eq!(nodes.spas.len(), spa.spa_tree.spas.len());
    }

    #[test]
    fn test_spa_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.reset_state();
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = LZ78SPA::<DirichletSPA>::from_bytes(&mut bytes).expect("from bytes failed");

        assert_eq!(spa.total_log_loss, new_spa.total_log_loss);
    }

    #[test]
    fn test_traverse_and_get_prob_lz78_spa() {
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");

        let pdf = spa
            .traverse_and_get_prob(&[0, 1])
            .expect("failed to traverse and get prob");
        assert_eq!(pdf.len(), 2);
        println!("pdf: {:?}", pdf);
    }
    #[test]
    fn test_traverse_and_get_prob_with_lookahead_lz78_spa() {
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        let input = BinarySequence::from_data(bitvec![1, 0].repeat(1000));
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");
        // input being [0,1] lookahead being [1,0]
        let pdf = spa
            .traverse_and_get_prob_with_lookahead(&[1, 0, 1], &[1, 0])
            .expect("failed to traverse and get prob with lookahead");
        assert_eq!(pdf.len(), 2);
        println!("pdf: {:?}", pdf);
    }
}
