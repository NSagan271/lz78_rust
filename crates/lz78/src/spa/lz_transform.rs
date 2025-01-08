use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

use crate::spa::util::adaptive_gamma;
use crate::storage::ToFromBytes;

use super::{
    generation::{gen_symbol_from_spa, GenerationParams, GenerationSPA},
    states::{LZ78EnsembleState, LZ78State, SPAState, LZ_ROOT_IDX},
    BackshiftParsing, Ensemble, LZ78SPAParams, SPAParams, SPA,
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
}

impl<S> SPATree<S> {
    pub fn new(params: &LZ78SPAParams) -> Result<Self>
    where
        S: SPA,
    {
        Ok(Self {
            spas: vec![S::new(&params.inner_params)?],
            branch_mappings: vec![HashMap::new()],
        })
    }

    fn prune(&mut self, min_count: u64)
    where
        S: SPA,
    {
        let mut curr_write_idx = 0;
        let mut old_idx_to_new_idx: HashMap<u64, u64> = HashMap::new();
        for i in 0..self.spas.len() {
            if self.spas[i].num_symbols_seen() < min_count {
                continue;
            }
            old_idx_to_new_idx.insert(i as u64, curr_write_idx);

            if curr_write_idx < i as u64 {
                self.spas.swap(i, curr_write_idx as usize);
                self.branch_mappings.swap(i, curr_write_idx as usize);
            }

            curr_write_idx += 1;
        }
        self.spas.truncate(curr_write_idx as usize);
        self.branch_mappings.truncate(curr_write_idx as usize);

        for mapping in self.branch_mappings.iter_mut() {
            let keys = mapping.keys().map(|x| *x).collect_vec();
            for k in keys {
                if !old_idx_to_new_idx.contains_key(&mapping[&k]) {
                    mapping.remove(&k);
                    continue;
                }
                mapping.insert(k, old_idx_to_new_idx[&mapping[&k]]);
            }
        }
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
        params: &mut LZ78SPAParams,
        sym: u32,
    ) -> Result<()>
    where
        S: SPA,
    {
        let prev_node = state.node;
        self.traverse_one_symbol_frozen(state, sym);
        if state.node == LZ_ROOT_IDX {
            // add a new leaf
            self.add_new_spa(prev_node, sym, S::new(&params.inner_params)?);
        }
        Ok(())
    }

    fn apply_adaptive_gamma(&self, state: &mut LZ78State, params: &mut LZ78SPAParams) -> Result<()>
    where
        S: SPA,
    {
        let node = state.node as usize;
        params.inner_params.maybe_set_gamma(adaptive_gamma(
            params.default_gamma,
            params.adaptive_gamma,
            state.depth,
            self.spas[node].num_symbols_seen() + 1,
        ));

        Ok(())
    }

    pub fn train_on_symbol(
        &mut self,
        state: &mut LZ78State,
        params: &mut LZ78SPAParams,
        sym: u32,
    ) -> Result<f64>
    where
        S: SPA,
    {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node].train_on_symbol(sym, &mut params.inner_params, inner_state)
    }

    pub fn test_on_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78SPAParams,
        sym: u32,
    ) -> Result<f64>
    where
        S: SPA,
    {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node as usize].test_on_symbol(sym, &mut params.inner_params, inner_state, None)
    }

    pub fn spa_for_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78SPAParams,
        sym: u32,
    ) -> Result<f64>
    where
        S: SPA,
    {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node as usize].spa_for_symbol(sym, &mut params.inner_params, inner_state, None)
    }

    pub fn spa(&self, state: &mut LZ78State, params: &mut LZ78SPAParams) -> Result<Vec<f64>>
    where
        S: SPA,
    {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node as usize].spa(&mut params.inner_params, inner_state, None)
    }
}

impl<S> ToFromBytes for SPATree<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
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

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
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

        Ok(Self {
            spas,
            branch_mappings,
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
    pub fn prune(&mut self, min_count: u64) {
        self.spa_tree.prune(min_count);
    }

    pub fn update_current_node_spa(
        &mut self,
        sym: u32,
        state: &mut LZ78State,
        params: &mut LZ78SPAParams,
    ) -> Result<f64> {
        let loss = self.spa_tree.train_on_symbol(state, params, sym)?;
        self.total_log_loss += loss;
        Ok(loss)
    }

    pub fn update_tree_structure(
        &mut self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut LZ78State,
    ) -> Result<()> {
        let params = params.try_get_lz78_mut()?;

        let prev_node = state.node;
        let prev_depth = state.depth;

        self.spa_tree
            .traverse_one_symbol_and_maybe_grow(state, params, sym)?;

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

    pub fn maybe_backshift_parse(
        &self,
        bs_parse_params: BackshiftParsing,
        reseeding_seq: &[u32],
        state: &mut LZ78State,
    ) -> Result<()> {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context

        let (min_spa_training_pts, desired_context_length) = if let BackshiftParsing::Enabled {
            desired_context_length,
            min_spa_training_points,
        } = bs_parse_params
        {
            (min_spa_training_points, desired_context_length)
        } else {
            return Ok(());
        };

        if self.spa_tree.spas[state.node as usize].num_symbols_seen() < min_spa_training_pts {
            let reseeding_start = reseeding_seq.len()
                - (state.depth as usize - 1).min(desired_context_length as usize);
            let reseeding_end = reseeding_seq.len();

            // keep on trying to re-seed the SPA
            for start_idx in reseeding_start..reseeding_end {
                state.go_to_root();

                for idx in start_idx..reseeding_end {
                    let sym = reseeding_seq[idx];
                    self.spa_tree.traverse_one_symbol_frozen(state, sym);
                    if state.node == LZ_ROOT_IDX {
                        break;
                    }
                }

                // re-seeding was successful!
                if state.node != LZ_ROOT_IDX
                    && self.spa_tree.spas[state.node as usize].num_symbols_seen()
                        >= min_spa_training_pts
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

    fn get_ensemble_spa(
        &self,
        states: &mut LZ78EnsembleState,
        params: &mut LZ78SPAParams,
        context_syms: &[u32],
    ) -> Result<Vec<f64>> {
        let states = &mut states.states;
        let bs_parse = params.backshift_parsing;
        if states.len() == 1 {
            self.maybe_backshift_parse(bs_parse, context_syms, &mut states[0])?;
            return self.spa_tree.spa(&mut states[0], params);
        }
        let mut spas = Vec::with_capacity(states.len() * params.alphabet_size as usize);
        let depths = Array1::from_vec(states.iter().map(|s| s.depth as f64).collect_vec());
        for state in states.iter_mut() {
            self.maybe_backshift_parse(bs_parse, &context_syms, state)?;
            spas.extend(self.spa_tree.spa(state, params)?);
        }
        let spas = Array2::from_shape_vec((states.len(), params.alphabet_size as usize), spas)?;

        return Ok(match params.ensemble {
            Ensemble::Average(_) => spas.mean_axis(Axis(0)).unwrap(),
            Ensemble::Entropy(_) => {
                let entropy = -(spas.clone() * spas.clone().log2()).sum_axis(Axis(1));
                let ent_min = *entropy.iter().min_by(|&x, &y| x.total_cmp(y)).unwrap();
                let ent_max = *entropy.iter().max_by(|&x, &y| x.total_cmp(y)).unwrap();
                let norm_ent = (entropy - ent_min) / (ent_max - ent_min + 1e-10);
                let mut weights = (-norm_ent / 2.0).exp();
                weights /= weights.sum();

                (spas.reversed_axes() * weights)
                    .reversed_axes()
                    .sum_axis(Axis(0))
            }
            Ensemble::Depth(_) => {
                let dep_min = *depths.iter().min_by(|&x, &y| x.total_cmp(y)).unwrap();
                let dep_max = *depths.iter().max_by(|&x, &y| x.total_cmp(y)).unwrap();

                let norm_dep = (depths - dep_min) / (dep_max - dep_min + 1e-6);
                let mut weights = norm_dep.exp();
                weights /= weights.sum();

                (spas.reversed_axes() * weights)
                    .reversed_axes()
                    .sum_axis(Axis(0))
            }
            Ensemble::None => bail!("ensemble disabled but multiple states created"),
        }
        .to_vec());
    }

    fn traverse_and_maybe_grow_ensemble(&self, states: &mut LZ78EnsembleState, sym: u32) {
        for state in &mut states.states {
            self.spa_tree.traverse_one_symbol_frozen(state, sym);
        }
        if states.states.len() < states.max_size as usize {
            states.states.push(states.base_state.clone());
        }
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPA,
{
    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;
        let loss = self.update_current_node_spa(input, state, params.try_get_lz78_mut()?)?;
        self.update_tree_structure(input, params, state)?;

        Ok(loss)
    }

    fn spa_for_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            return Ok(self.get_ensemble_spa(
                state.try_get_ensemble()?,
                params,
                context_syms.unwrap_or(&[]),
            )?[sym as usize]);
        }
        self.spa_tree
            .spa_for_symbol(state.try_get_lz78()?, params, sym)
    }

    fn spa(
        &self,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Vec<f64>> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            return self.get_ensemble_spa(
                state.try_get_ensemble()?,
                params,
                context_syms.unwrap_or(&[]),
            );
        }

        self.spa_tree.spa(state.try_get_lz78()?, params)
    }

    fn test_on_symbol(
        &self,
        input: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            let states = state.try_get_ensemble()?;
            let spa = self.get_ensemble_spa(states, params, context_syms.unwrap_or(&[]))?;
            let loss = -spa[input as usize].log2();
            self.traverse_and_maybe_grow_ensemble(states, input);
            return Ok(loss);
        }

        let state = state.try_get_lz78()?;
        if let Some(ctx) = context_syms {
            self.maybe_backshift_parse(params.backshift_parsing, ctx, state)?;
        }
        let loss = self.spa_tree.test_on_symbol(state, params, input)?;
        self.spa_tree.traverse_one_symbol_frozen(state, input);

        Ok(loss)
    }

    fn new(params: &SPAParams) -> Result<Self> {
        Ok(Self {
            spa_tree: SPATree::new(params.try_get_lz78()?)?,
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
        params: &mut LZ78SPAParams,
    ) -> Result<f64> {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node as usize].input_seed_data_symbol(sym, &mut params.inner_params, inner_state)
    }

    pub fn generate_one_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78SPAParams,
        gen_params: &GenerationParams,
        context_syms: &[u32],

        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        let node = state.node as usize;
        self.apply_adaptive_gamma(state, params)?;
        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        self.spas[node as usize].generate_one_symbol(
            rng_sample,
            &mut params.inner_params,
            gen_params,
            inner_state,
            context_syms,
        )
    }
}

impl<S> GenerationSPA for LZ78SPA<S>
where
    S: GenerationSPA,
{
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            let states = state.try_get_ensemble()?;
            let loss = -self.get_ensemble_spa(states, params, &[])?[sym as usize].log2();
            for state in states.states.iter_mut() {
                self.spa_tree.input_seed_data_symbol(state, sym, params)?;
            }
            self.traverse_and_maybe_grow_ensemble(states, sym);
            return Ok(loss);
        }
        let state = state.try_get_lz78()?;
        let loss = self.spa_tree.input_seed_data_symbol(state, sym, params)?;
        self.spa_tree.traverse_one_symbol_frozen(state, sym);

        Ok(loss)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &mut SPAParams,
        gen_params: &GenerationParams,
        state: &mut SPAState,
        context_syms: &[u32],
    ) -> Result<(u32, f64)> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            let states = state.try_get_ensemble()?;
            let spa = self.get_ensemble_spa(states, params, context_syms)?;
            let (new_sym, sym_loss) = gen_symbol_from_spa(rng_sample, gen_params, &spa)?;
            self.traverse_and_maybe_grow_ensemble(states, new_sym);
            return Ok((new_sym, sym_loss));
        }

        let state = state.try_get_lz78()?;
        let bs_parse = params.backshift_parsing;
        self.maybe_backshift_parse(bs_parse, context_syms, state)?;

        let (new_sym, sym_loss) = self.spa_tree.generate_one_symbol(
            state,
            params,
            gen_params,
            context_syms,
            rng_sample,
        )?;
        self.spa_tree.traverse_one_symbol_frozen(state, new_sym);

        Ok((new_sym, sym_loss))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sequence::BinarySequence,
        spa::{
            basic_spas::DirichletSPA, lz_transform::LZ78SPA, util::LbAndTemp, AdaptiveGamma,
            Ensemble,
        },
    };
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut params = SPAParams::new_lz78_dirichlet(
            2,
            0.5,
            LbAndTemp::Skip,
            AdaptiveGamma::None,
            Ensemble::None,
            BackshiftParsing::Enabled {
                desired_context_length: 2,
                min_spa_training_points: 1,
            },
            false,
        );
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        state.reset();
        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        state.reset();
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn sanity_check_ensemble() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(1000));
        let mut params = SPAParams::new_lz78_dirichlet(
            2,
            0.5,
            LbAndTemp::Skip,
            AdaptiveGamma::None,
            Ensemble::None,
            BackshiftParsing::Enabled {
                desired_context_length: 2,
                min_spa_training_points: 2,
            },
            false,
        );

        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        state.reset();
        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        let mut params = SPAParams::new_lz78_dirichlet(
            2,
            0.5,
            LbAndTemp::Skip,
            AdaptiveGamma::None,
            Ensemble::Entropy(1),
            BackshiftParsing::Enabled {
                desired_context_length: 2,
                min_spa_training_points: 2,
            },
            false,
        );
        let mut state = params.get_new_state();
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        assert!((loss1 - loss2).abs() < 1e-4);

        let mut params = SPAParams::new_lz78_dirichlet(
            2,
            0.5,
            LbAndTemp::Skip,
            AdaptiveGamma::None,
            Ensemble::Depth(3),
            BackshiftParsing::Enabled {
                desired_context_length: 2,
                min_spa_training_points: 2,
            },
            false,
        );
        let mut state = params.get_new_state();
        let loss3 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        println!("without ensemble = {loss1}, with ensemble = {loss3}")
    }

    #[test]
    fn test_lz_transformed_nodes_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut params = SPAParams::default_lz78_dirichlet(2);
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
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
        let mut params = SPAParams::default_lz78_dirichlet(2);
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = LZ78SPA::<DirichletSPA>::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.total_log_loss, new_spa.total_log_loss);
    }

    #[test]
    fn test_pruning() {
        let input = BinarySequence::from_data(bitvec![0, 1, 1, 0, 1, 1, 1].repeat(500));
        let mut params = SPAParams::default_lz78_dirichlet(2);
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let old_spas = spa.spa_tree.spas.clone();
        let old_mappings = spa.spa_tree.branch_mappings.clone();

        let min_count = 5;
        spa.prune(min_count);

        println!(
            "Old num nodes: {}, new: {}",
            old_spas.len(),
            spa.spa_tree.spas.len()
        );
        assert!(spa.spa_tree.spas.len() < old_spas.len());

        // traverse the original and pruned trees to check
        let mut old_stack = vec![0u64];
        let mut new_stack = vec![0u64];
        while old_stack.len() > 0 {
            let old_node = old_stack.pop().unwrap();
            let old_count = old_spas[old_node as usize].num_symbols_seen();
            if old_count < min_count {
                continue;
            }

            assert!(new_stack.len() > 0);
            let new_node = new_stack.pop().unwrap();

            assert_eq!(
                old_count,
                spa.spa_tree.spas[new_node as usize].num_symbols_seen()
            );

            old_stack.extend(
                old_mappings[old_node as usize]
                    .iter()
                    .sorted_by_key(|(k, _)| **k)
                    .map(|(_, v)| *v),
            );

            new_stack.extend(
                spa.spa_tree.branch_mappings[new_node as usize]
                    .iter()
                    .sorted_by_key(|(k, _)| **k)
                    .map(|(_, v)| *v),
            );
        }
    }
}
