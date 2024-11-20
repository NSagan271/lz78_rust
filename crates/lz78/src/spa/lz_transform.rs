use anyhow::{anyhow, bail, Result};
use bitvec::vec::BitVec;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{storage::ToFromBytes, util::sample_from_pdf};

use super::{
    generation::{GenerationParams, GenerationSPA},
    SPAParams, SPA,
};

/// All nodes in the LZ78 tree are stored in the `spas` array, and the tree
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
pub struct SPATree<S> {
    pub spas: Vec<S>,
    pub branch_mappings: Vec<HashMap<u32, u64>>,
    pub pending_reset: BitVec<u64>,
    pub params: Arc<SPAParams>,
}

impl<S> SPATree<S> {
    pub const ROOT_IDX: u64 = 0;

    pub fn new(params: Arc<SPAParams>) -> Result<Self>
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

    pub fn traverse_one_symbol_frozen(&self, state: u64, sym: u32) -> u64 {
        if self.branch_mappings[state as usize].contains_key(&sym) {
            self.branch_mappings[state as usize][&sym]
        } else {
            Self::ROOT_IDX
        }
    }

    pub fn add_new_spa(&mut self, state: u64, sym: u32, new_spa: S) {
        let new_node_idx = self.spas.len() as u64;
        self.spas.push(new_spa);
        self.branch_mappings[state as usize].insert(sym, new_node_idx);
        self.branch_mappings.push(HashMap::new());
        self.pending_reset.push(false);
    }

    pub fn traverse_one_symbol_and_maybe_grow(&mut self, state: u64, sym: u32) -> Result<u64>
    where
        S: SPA,
    {
        let new_state = self.traverse_one_symbol_frozen(state, sym);
        if new_state == Self::ROOT_IDX {
            // add a new leaf
            self.add_new_spa(state, sym, S::new(&self.params)?);
        }

        Ok(new_state)
    }

    pub fn train_on_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].train_on_symbol(sym, &self.params)
    }

    pub fn test_on_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].test_on_symbol(sym, &self.params)
    }

    pub fn spa_for_symbol(&mut self, state: u64, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        if self.pending_reset[state as usize] {
            self.spas[state as usize].reset_state();
            self.pending_reset.set(state as usize, false);
        }
        self.spas[state as usize].spa_for_symbol(sym, &self.params)
    }

    pub fn spa(&mut self, state: u64) -> Result<Vec<f64>>
    where
        S: SPA,
    {
        let mut result: Vec<f64> = Vec::with_capacity(self.params.alphabet_size() as usize);
        for sym in 0..self.params.alphabet_size() {
            result.push(self.spa_for_symbol(state, sym)?);
        }

        Ok(result)
    }

    pub fn recursively_reset_state(&mut self) {
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

pub struct LZ78GenerationState {
    tree_state: u64,
    nodes_seen: HashSet<u64>,
    reseeding_seq: Vec<u32>,
    last_time_root_seen: u64,
}

impl LZ78GenerationState {
    pub fn new() -> Self {
        Self {
            tree_state: 0,
            nodes_seen: HashSet::new(),
            reseeding_seq: Vec::new(),
            last_time_root_seen: 0,
        }
    }

    pub fn clear(&mut self) {
        self.tree_state = 0;
        self.nodes_seen.clear();
        self.reseeding_seq.clear();
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
    pub fn new() -> Self {
        Self {
            max_depth: 0,
            current_depth: 0,
            deepest_leaf: 0,
            leaf_depths: HashMap::new(),
            parent_and_sym_map: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.leaf_depths.clear();
        self.parent_and_sym_map.clear();
    }

    pub fn add_leaf(&mut self, state: u64, new_leaf_idx: u64, new_leaf_sym: u32, leaf_depth: u32) {
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
                rand_iter.next().ok_or_else(|| {
                    let var_name = anyhow!("could not get sample from rng");
                    var_name
                })?,
            ) as u32;
            state = spa.spa_tree.traverse_one_symbol_frozen(state, sym);
        }
        results.push(depth - 1);
    }

    Ok(results)
}
/// LZ78 implementation of the sequential probability assignment
pub struct LZ78SPA<S> {
    pub spa_tree: SPATree<S>,
    pub state: u64,
    n: u64,
    total_log_loss: f64,
    pub gen_state: LZ78GenerationState,
    pub debug: LZ78DebugState,
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
}

/// Decoupling the  training of the current node's SPA and the tree traversal
/// / adding a new node will be helpful when building a causally-processed SPA.
impl<S> LZ78SPA<S>
where
    S: SPA,
{
    pub fn update_current_node_spa(&mut self, sym: u32) -> Result<f64> {
        let loss = self.spa_tree.train_on_symbol(self.state, sym)?;
        self.total_log_loss += loss;
        Ok(loss)
    }

    pub fn update_tree_structure(&mut self, sym: u32, params: &SPAParams) -> Result<()> {
        let params = if let SPAParams::LZ78(p) = params {
            p
        } else {
            bail!("Invalid SPAParams for LZ78 SPA")
        };

        let old_state = self.state;

        self.state = self
            .spa_tree
            .traverse_one_symbol_and_maybe_grow(self.state, sym)?;

        if params.debug {
            if self.state == SPATree::<S>::ROOT_IDX {
                self.debug.add_leaf(
                    old_state,
                    self.spa_tree.spas.len() as u64 - 1,
                    sym,
                    self.debug.current_depth,
                );
                self.debug.current_depth = 0;
            } else {
                self.debug.current_depth += 1;
            }
        }

        self.n += 1;

        Ok(())
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPA,
{
    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let loss = self.update_current_node_spa(input)?;
        self.update_tree_structure(input, params)?;

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

impl<S> SPATree<S>
where
    S: GenerationSPA,
{
    pub fn input_seed_data_symbol(
        &mut self,
        state: u64,
        sym: u32,
        _params: &SPAParams,
    ) -> Result<f64> {
        self.spas[state as usize].input_seed_data_symbol(sym, &self.params)
    }

    pub fn generate_one_symbol(
        &mut self,
        state: u64,
        gen_params: &GenerationParams,
        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        self.spas[state as usize].generate_one_symbol(rng_sample, &self.params, gen_params)
    }
}

impl<S> LZ78SPA<S>
where
    S: GenerationSPA,
{
    pub fn maybe_reseed_tree(&mut self, gen_params: &GenerationParams) {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context
        if self.gen_state.tree_state == SPATree::<S>::ROOT_IDX
            || self.spa_tree.spas[self.gen_state.tree_state as usize].num_symbols_seen()
                < gen_params.min_spa_training_points
        {
            // keep on trying to re-seed the SPA
            for start_idx in (self.gen_state.last_time_root_seen + 1).max(
                self.gen_state.reseeding_seq.len() as u64
                    - gen_params
                        .desired_context_length
                        .min(self.gen_state.reseeding_seq.len() as u64),
            )..(self.gen_state.reseeding_seq.len() as u64)
            {
                self.gen_state.last_time_root_seen = start_idx;
                self.gen_state.tree_state = SPATree::<S>::ROOT_IDX;
                for &sym in self.gen_state.reseeding_seq.iter().skip(start_idx as usize) {
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
            self.gen_state.last_time_root_seen = self.gen_state.reseeding_seq.len() as u64;
        }
    }

    pub fn update_gen_state_with_curr_node(&mut self) {
        if self.gen_state.tree_state == SPATree::<S>::ROOT_IDX {
            self.gen_state.last_time_root_seen = self.gen_state.reseeding_seq.len() as u64;
        }

        self.gen_state.nodes_seen.insert(self.gen_state.tree_state);
    }

    /// Used by the causally processed SPA
    pub fn get_gen_tree_state(&self) -> u64 {
        self.gen_state.tree_state
    }

    /// Used by the causally processed SPA
    pub fn update_gen_state_with_sym(&mut self, sym: u32) {
        self.gen_state.reseeding_seq.push(sym);
    }

    pub fn traverse_tree_generation(&mut self, sym: u32) {
        self.gen_state.tree_state = self
            .spa_tree
            .traverse_one_symbol_frozen(self.gen_state.tree_state, sym);
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
        self.update_gen_state_with_curr_node();

        let loss = self
            .spa_tree
            .input_seed_data_symbol(self.gen_state.tree_state, sym, params)?;
        self.gen_state.reseeding_seq.push(sym);

        self.traverse_tree_generation(sym);

        Ok(loss)
    }

    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        _params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, f64)> {
        self.maybe_reseed_tree(gen_params);
        self.update_gen_state_with_curr_node();

        let (new_sym, sym_loss) =
            self.spa_tree
                .generate_one_symbol(self.gen_state.tree_state, gen_params, rng_sample)?;
        self.gen_state.reseeding_seq.push(new_sym);
        self.traverse_tree_generation(new_sym);
        Ok((new_sym, sym_loss))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sequence::BinarySequence,
        spa::{basic_spas::DirichletSPA, lz_transform::LZ78SPA},
    };
    use bitvec::prelude::*;

    use super::*;

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
}
