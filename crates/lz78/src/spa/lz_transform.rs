use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use std::{collections::HashMap, sync::Arc};

use crate::storage::ToFromBytes;

use super::{
    generation::{GenerationParams, GenerationSPA},
    states::{LZ78State, SPAState, LZ_ROOT_IDX},
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
    pub params: Arc<SPAParams>,
}

impl<S> SPATree<S> {
    pub fn new(params: Arc<SPAParams>) -> Result<Self>
    where
        S: SPA,
    {
        Ok(Self {
            spas: vec![S::new(&params)?],
            branch_mappings: vec![HashMap::new()],
            params,
        })
    }

    pub fn traverse_one_symbol_frozen(&self, state: &mut LZ78State, sym: u32) {
        // println!("{} {}", state.depth, sym);
        if self.branch_mappings[state.node as usize].contains_key(&sym) {
            state.node = self.branch_mappings[state.node as usize][&sym];
            state.depth += 1;
        } else {
            state.go_to_root();
        }
    }

    pub fn add_new_spa(&mut self, node: u64, sym: u32, new_spa: S) {
        let new_node_idx = self.spas.len() as u64;
        self.spas.push(new_spa);
        self.branch_mappings[node as usize].insert(sym, new_node_idx);
        self.branch_mappings.push(HashMap::new());
    }

    pub fn traverse_one_symbol_and_maybe_grow(
        &mut self,
        state: &mut LZ78State,
        sym: u32,
    ) -> Result<()>
    where
        S: SPA,
    {
        let prev_node = state.node;
        self.traverse_one_symbol_frozen(state, sym);
        if state.node == LZ_ROOT_IDX {
            // add a new leaf
            self.add_new_spa(prev_node, sym, S::new(&self.params)?);
        }
        Ok(())
    }

    pub fn train_on_symbol(&mut self, state: &mut LZ78State, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        self.spas[state.node as usize].train_on_symbol(
            sym,
            &self.params,
            state
                .get_child_state(&self.params)
                .unwrap_or(&mut SPAState::None),
        )
    }

    pub fn test_on_symbol(&self, state: &mut LZ78State, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        self.spas[state.node as usize].test_on_symbol(
            sym,
            &self.params,
            state
                .get_child_state(&self.params)
                .unwrap_or(&mut SPAState::None),
        )
    }

    pub fn spa_for_symbol(&self, state: &mut LZ78State, sym: u32) -> Result<f64>
    where
        S: SPA,
    {
        self.spas[state.node as usize].spa_for_symbol(
            sym,
            &self.params,
            state
                .get_child_state(&self.params)
                .unwrap_or(&mut SPAState::None),
        )
    }

    pub fn spa(&mut self, state: &mut LZ78State) -> Result<Vec<f64>>
    where
        S: SPA,
    {
        let mut result: Vec<f64> = Vec::with_capacity(self.params.alphabet_size() as usize);
        for sym in 0..self.params.alphabet_size() {
            result.push(self.spa_for_symbol(state, sym)?);
        }

        Ok(result)
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
        let params = SPAParams::from_bytes(bytes)?;

        Ok(Self {
            spas,
            branch_mappings,
            params: Arc::new(params),
        })
    }
}

/// Tracks, e.g., the depth of each leaf for better understanding the tree
/// built by the LZ78 SPA
#[derive(Debug, Clone)]
pub struct LZ78DebugState {
    pub max_depth: u32,
    pub deepest_leaf: u64,
    pub leaf_depths: HashMap<u64, u32>,
    pub parent_and_sym_map: HashMap<u64, (u64, u32)>,
    pub depths_traversed: Vec<u32>,
}

impl LZ78DebugState {
    pub fn new() -> Self {
        Self {
            max_depth: 0,
            deepest_leaf: 0,
            leaf_depths: HashMap::new(),
            parent_and_sym_map: HashMap::new(),
            depths_traversed: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.leaf_depths.clear();
        self.parent_and_sym_map.clear();
        self.depths_traversed.clear();
    }

    pub fn clear_depths_traversed(&mut self) {
        self.depths_traversed.clear();
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

        bytes.put_u64_le(self.depths_traversed.len() as u64);
        for &d in self.depths_traversed.iter() {
            bytes.put_u32_le(d);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let max_depth = bytes.get_u32_le();
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

        let n_syms = bytes.get_u64_le();
        let mut depths_traversed = Vec::with_capacity(n_syms as usize);
        for _ in 0..n_syms {
            depths_traversed.push(bytes.get_u32_le());
        }

        Ok(Self {
            max_depth,
            leaf_depths,
            parent_and_sym_map,
            depths_traversed,
            deepest_leaf,
        })
    }
}

/// LZ78 implementation of the sequential probability assignment
pub struct LZ78SPA<S> {
    pub spa_tree: SPATree<S>,
    n: u64,
    total_log_loss: f64,
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
    pub fn update_current_node_spa(&mut self, sym: u32, state: &mut LZ78State) -> Result<f64> {
        let loss = self.spa_tree.train_on_symbol(state, sym)?;
        self.total_log_loss += loss;
        Ok(loss)
    }

    pub fn update_tree_structure(
        &mut self,
        sym: u32,
        params: &SPAParams,
        state: &mut LZ78State,
    ) -> Result<()> {
        let params = params.try_get_lz78()?;

        let prev_node = state.node;
        let prev_depth = state.depth;

        self.spa_tree
            .traverse_one_symbol_and_maybe_grow(state, sym)?;

        if params.debug && state.node == LZ_ROOT_IDX {
            self.debug.add_leaf(
                prev_node,
                self.spa_tree.spas.len() as u64 - 1,
                sym,
                prev_depth + 1,
            );
        }

        self.n += 1;

        Ok(())
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPA,
{
    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;
        let loss = self.update_current_node_spa(input, state)?;
        self.update_tree_structure(input, params, state)?;

        Ok(loss)
    }

    fn spa_for_symbol(&self, sym: u32, _params: &SPAParams, state: &mut SPAState) -> Result<f64> {
        self.spa_tree.spa_for_symbol(state.try_get_lz78()?, sym)
    }

    fn test_on_symbol(&self, input: u32, _params: &SPAParams, state: &mut SPAState) -> Result<f64> {
        let state = state.try_get_lz78()?;
        let loss = self.spa_tree.test_on_symbol(state, input)?;
        self.spa_tree.traverse_one_symbol_frozen(state, input);

        Ok(loss)
    }

    fn new(params: &SPAParams) -> Result<Self> {
        let params = params.try_get_lz78()?.inner_params.clone();
        Ok(Self {
            spa_tree: SPATree::new(params)?,
            n: 0,
            total_log_loss: 0.0,
            debug: LZ78DebugState::new(),
        })
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
        let n = bytes.get_u64_le();
        let total_log_loss = bytes.get_f64_le();
        let debug = LZ78DebugState::from_bytes(bytes)?;

        Ok(Self {
            spa_tree,
            n,
            total_log_loss,
            debug,
        })
    }
}

impl<S> SPATree<S>
where
    S: GenerationSPA,
{
    pub fn input_seed_data_symbol(
        &self,
        state: &mut LZ78State,
        sym: u32,
        _params: &SPAParams,
    ) -> Result<f64> {
        self.spas[state.node as usize].input_seed_data_symbol(
            sym,
            &self.params,
            state
                .get_child_state(&self.params)
                .unwrap_or(&mut SPAState::None),
        )
    }

    pub fn generate_one_symbol(
        &self,
        state: &mut LZ78State,
        gen_params: &GenerationParams,
        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        self.spas[state.node as usize].generate_one_symbol(
            rng_sample,
            &self.params,
            gen_params,
            state
                .get_child_state(&self.params)
                .unwrap_or(&mut SPAState::None),
        )
    }
}

impl<S> LZ78SPA<S>
where
    S: GenerationSPA,
{
    pub fn maybe_reseed_tree(
        &self,
        gen_params: &GenerationParams,
        state: &mut LZ78State,
    ) -> Result<()> {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context

        if self.spa_tree.spas[state.node as usize].num_symbols_seen()
            < gen_params.min_spa_training_points
        {
            let reseeding_start = state.gen_state.reseeding_seq.len()
                - (state.depth as usize - 1).min(gen_params.desired_context_length as usize);
            let reseeding_end = state.gen_state.reseeding_seq.len();

            // keep on trying to re-seed the SPA
            for start_idx in reseeding_start..reseeding_end {
                state.go_to_root();

                for idx in start_idx..reseeding_end {
                    let sym = state.gen_state.reseeding_seq[idx];
                    self.spa_tree.traverse_one_symbol_frozen(state, sym);
                    if state.node == LZ_ROOT_IDX {
                        break;
                    }
                }

                // re-seeding was successful!
                if state.node != LZ_ROOT_IDX
                    && self.spa_tree.spas[state.node as usize].num_symbols_seen()
                        >= gen_params.min_spa_training_points
                {
                    break;
                }
            }
        }
        // if reseeding failed, we don't want to end up at a leaf!
        if self.spa_tree.spas[state.node as usize].num_symbols_seen() == 0 {
            state.go_to_root();
        }

        Ok(())
    }

    pub fn traverse_tree_generation(&self, sym: u32, state: &mut LZ78State) {
        self.spa_tree.traverse_one_symbol_frozen(state, sym);
    }
}

impl<S> GenerationSPA for LZ78SPA<S>
where
    S: GenerationSPA,
{
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;
        state.update_gen_state_with_curr_node();

        let loss = self.spa_tree.input_seed_data_symbol(state, sym, params)?;
        state.update_gen_state_with_sym(sym);
        self.traverse_tree_generation(sym, state);

        Ok(loss)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        _params: &SPAParams,
        gen_params: &GenerationParams,
        state: &mut SPAState,
    ) -> Result<(u32, f64)> {
        let state = state.try_get_lz78()?;
        self.maybe_reseed_tree(gen_params, state)?;
        state.update_gen_state_with_curr_node();

        let (new_sym, sym_loss) = self
            .spa_tree
            .generate_one_symbol(state, gen_params, rng_sample)?;
        state.update_gen_state_with_sym(new_sym);
        self.traverse_tree_generation(new_sym, state);
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
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &params, &mut state)
            .expect("failed to train spa");

        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &params,
                &mut state,
            )
            .expect("failed to compute test loss");
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                &params,
                &mut state,
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_lz_transformed_nodes_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let params = SPAParams::new_lz78_dirichlet(2, 0.5, false);
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &params, &mut state)
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
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &params, &mut state)
            .expect("failed to train spa");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = LZ78SPA::<DirichletSPA>::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.total_log_loss, new_spa.total_log_loss);
    }
}
