use crate::storage::ToFromBytes;

use super::{
    generation::{gen_symbol_from_spa, GenerationSPA, GenerationSPATree},
    params::{BackshiftParsing, Ensemble, LZ78Params, SPAParams},
    states::{LZ78EnsembleState, LZ78State, SPAState, LZ_ROOT_IDX},
    util::adaptive_gamma,
    SPATree, SPA,
};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use ndarray::{Array1, Array2, Axis};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub struct LZ78Tree<S> {
    pub spa_tree: S,
}

impl<S> ToFromBytes for LZ78Tree<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        self.spa_tree.to_bytes()
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            spa_tree: S::from_bytes(bytes)?,
        })
    }
}

impl<S> LZ78Tree<S>
where
    S: SPATree,
{
    pub fn new(params: &LZ78Params) -> Result<Self> {
        Ok(Self {
            spa_tree: S::new(&params.inner_params)?,
        })
    }

    pub fn traverse_one_symbol_frozen(&self, state: &mut LZ78State, sym: u32) {
        match self.spa_tree.get_child_idx(state.node, sym) {
            Some(idx) => {
                state.node = *idx;
                state.depth += 1;
            }
            None => {
                state.go_to_root();
            }
        }
    }

    pub fn traverse_one_symbol_and_maybe_grow(
        &mut self,
        state: &mut LZ78State,
        params: &mut LZ78Params,
        sym: u32,
    ) -> Result<()> {
        let prev_node = state.node;
        self.traverse_one_symbol_frozen(state, sym);
        if state.node == LZ_ROOT_IDX {
            // add a new leaf
            self.spa_tree
                .add_new(&params.inner_params, prev_node, sym)?;
        }
        Ok(())
    }

    fn apply_adaptive_gamma(&self, state: &mut LZ78State, params: &mut LZ78Params) {
        if let Some(gamma) = params.inner_params.maybe_get_gamma() {
            params.inner_params.maybe_set_gamma(adaptive_gamma(
                gamma,
                params.adaptive_gamma,
                state.depth,
                self.spa_tree.num_symbols_seen(state.node) + 1,
            ));
        }
    }

    pub fn train_on_symbol(
        &mut self,
        state: &mut LZ78State,
        params: &mut LZ78Params,
        sym: u32,
    ) -> Result<f64> {
        let old_gamma = params.inner_params.maybe_get_gamma();
        let node = state.node;
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);
        let loss =
            self.spa_tree
                .train_spa_on_symbol(node, sym, &mut params.inner_params, inner_state)?;
        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn test_on_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78Params,
        sym: u32,
    ) -> Result<f64> {
        let old_gamma = params.inner_params.maybe_get_gamma();
        let node = state.node;
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        let loss =
            self.spa_tree
                .test_on_symbol(node, sym, &mut params.inner_params, inner_state, None)?;
        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn spa_for_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78Params,
        sym: u32,
    ) -> Result<f64> {
        let node = state.node;
        let old_gamma = params.inner_params.maybe_get_gamma();
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        let loss =
            self.spa_tree
                .spa_for_symbol(node, sym, &mut params.inner_params, inner_state, None)?;

        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn spa(&self, state: &mut LZ78State, params: &mut LZ78Params) -> Result<Array1<f64>> {
        let node = state.node;
        let old_gamma = params.inner_params.maybe_get_gamma();
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        let loss = self
            .spa_tree
            .spa(node, &mut params.inner_params, inner_state, None)?;

        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }
}

impl<S> LZ78Tree<S>
where
    S: GenerationSPATree,
{
    pub fn input_seed_data_symbol(
        &self,
        state: &mut LZ78State,
        sym: u32,
        params: &mut LZ78Params,
    ) -> Result<f64> {
        let node = state.node;
        let old_gamma = params.inner_params.maybe_get_gamma();
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);
        let loss = self.spa_tree.input_seed_data_symbol(
            node,
            sym,
            &mut params.inner_params,
            inner_state,
        )?;

        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn generate_one_symbol(
        &self,
        state: &mut LZ78State,
        params: &mut LZ78Params,
        context_syms: &[u32],
        rng_sample: f64,
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        let node = state.node;

        let old_gamma = params.inner_params.maybe_get_gamma();
        self.apply_adaptive_gamma(state, params);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut params.inner_params)
            .unwrap_or(&mut none_state);

        let res = self.spa_tree.generate_one_symbol(
            node,
            rng_sample,
            &mut params.inner_params,
            inner_state,
            context_syms,
            temperature,
            topk,
        )?;

        if let Some(gamma) = old_gamma {
            params.inner_params.maybe_set_gamma(gamma);
        }
        Ok(res)
    }
}

pub struct LZ78SPA<S> {
    pub lz_tree: LZ78Tree<S>,
    n: u64,
    total_log_loss: f64,
}

impl<S> ToFromBytes for LZ78SPA<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.lz_tree.to_bytes()?;
        bytes.put_u64_le(self.n);
        bytes.put_f64_le(self.total_log_loss);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let lz_tree = LZ78Tree::<S>::from_bytes(bytes)?;
        let n = bytes.get_u64_le();
        let total_log_loss = bytes.get_f64_le();
        Ok(Self {
            lz_tree,
            n,
            total_log_loss,
        })
    }
}

impl<S> LZ78SPA<S>
where
    S: SPATree,
{
    pub fn get_normalized_log_loss(&self) -> f64 {
        self.total_log_loss / self.n as f64
    }

    pub fn prune(&mut self, min_count: u64) {
        self.lz_tree.spa_tree.prune(min_count);
    }

    pub fn update_current_node_spa(
        &mut self,
        sym: u32,
        state: &mut LZ78State,
        params: &mut LZ78Params,
    ) -> Result<f64> {
        let loss = self.lz_tree.train_on_symbol(state, params, sym)?;
        self.total_log_loss += loss;
        Ok(loss)
    }

    pub fn update_tree_structure(
        &mut self,
        sym: u32,
        params: &mut LZ78Params,
        state: &mut LZ78State,
    ) -> Result<()> {
        self.lz_tree
            .traverse_one_symbol_and_maybe_grow(state, params, sym)?;
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

        if self.lz_tree.spa_tree.num_symbols_seen(state.node) < min_spa_training_pts {
            let reseeding_start =
                reseeding_seq.len() - (reseeding_seq.len()).min(desired_context_length as usize);
            let reseeding_end = reseeding_seq.len();

            // keep on trying to re-seed the SPA
            for start_idx in reseeding_start..reseeding_end {
                state.go_to_root();

                for idx in start_idx..reseeding_end {
                    self.lz_tree
                        .traverse_one_symbol_frozen(state, reseeding_seq[idx]);
                    if state.node == LZ_ROOT_IDX {
                        break;
                    }
                }

                // re-seeding was successful!
                if state.node != LZ_ROOT_IDX
                    && self.lz_tree.spa_tree.num_symbols_seen(state.node) >= min_spa_training_pts
                {
                    break;
                }
            }
        }
        // if reseeding failed, we don't want to end up at a leaf!
        if self.lz_tree.spa_tree.num_symbols_seen(state.node) == 0 {
            state.go_to_root();
        }

        Ok(())
    }

    fn ensemble_spa_from_spas(
        &self,
        spas: Array2<f64>,
        depths: Array1<f64>,
        ensemble_type: Ensemble,
    ) -> Result<Array1<f64>> {
        Ok(match ensemble_type {
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
        })
    }

    fn get_ensemble_spa_par(
        &self,
        states: &mut LZ78EnsembleState,
        params: &mut LZ78Params,
        context_syms: &[u32],
    ) -> Result<Array1<f64>> {
        let state_vec = &mut states.states;
        let bs_parse = params.backshift_parsing;
        if state_vec.len() == 1 {
            self.maybe_backshift_parse(bs_parse, context_syms, &mut state_vec[0])?;
            return self.lz_tree.spa(&mut state_vec[0], params);
        }
        let mut depths = Vec::with_capacity(state_vec.len());
        let mut spas = Vec::with_capacity(state_vec.len());
        states.pool.as_mut().unwrap().install(|| {
            state_vec
                .par_iter_mut()
                .map(|state| {
                    self.maybe_backshift_parse(bs_parse, context_syms, state)
                        .unwrap();
                    let spa = self.lz_tree.spa(state, &mut params.clone()).unwrap();
                    (state.depth as f64, spa.to_vec())
                })
                .unzip_into_vecs(&mut depths, &mut spas);
        });
        let depths = Array1::from_vec(depths);
        let spas = Array2::from_shape_vec(
            (
                state_vec.len(),
                params.inner_params.alphabet_size() as usize,
            ),
            spas.concat(),
        )?;

        self.ensemble_spa_from_spas(spas, depths, params.ensemble)
    }

    fn get_ensemble_spa(
        &self,
        states: &mut LZ78EnsembleState,
        params: &mut LZ78Params,
        context_syms: &[u32],
    ) -> Result<Array1<f64>> {
        if states.is_parallel() {
            return self.get_ensemble_spa_par(states, params, context_syms);
        }
        let state_vec = &mut states.states;
        let bs_parse = params.backshift_parsing;
        if state_vec.len() == 1 {
            self.maybe_backshift_parse(bs_parse, context_syms, &mut state_vec[0])?;
            return self.lz_tree.spa(&mut state_vec[0], params);
        }

        let mut spas = Array2::zeros((
            state_vec.len(),
            params.inner_params.alphabet_size() as usize,
        ));
        let mut depths: Array1<f64> = Array1::zeros(state_vec.len());

        for (i, state) in state_vec.iter_mut().enumerate() {
            self.maybe_backshift_parse(bs_parse, context_syms, state)
                .unwrap();
            let mut row = spas.index_axis_mut(Axis(0), i);
            row += &self.lz_tree.spa(state, &mut params.clone()).unwrap();
            self.lz_tree.spa(state, &mut params.clone()).unwrap();
            depths[i] = state.depth as f64;
        }

        self.ensemble_spa_from_spas(spas, depths, params.ensemble)
    }

    fn traverse_and_maybe_grow_ensemble(&self, states: &mut LZ78EnsembleState, sym: u32) {
        if states.is_parallel() {
            states.pool.as_mut().unwrap().install(|| {
                states.states.par_iter_mut().for_each(|state| {
                    self.lz_tree.traverse_one_symbol_frozen(state, sym);
                });
            })
        } else {
            for state in states.states.iter_mut() {
                self.lz_tree.traverse_one_symbol_frozen(state, sym);
            }
        }
        if states.states.len() < states.max_size as usize {
            states.states.push(states.base_state.clone());
        }
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPATree,
{
    fn train_on_symbol(
        &mut self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let params = params.try_get_lz78_mut()?;
        let state = state.try_get_lz78()?;
        let loss = self.update_current_node_spa(sym, state, params)?;
        self.update_tree_structure(sym, params, state)?;

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
            let state = state.try_get_ensemble()?;
            return Ok(
                self.get_ensemble_spa(state, params, context_syms.unwrap_or(&[]))?[sym as usize],
            );
        }

        let state = state.try_get_lz78()?;
        if let Some(ctx) = context_syms {
            self.maybe_backshift_parse(params.backshift_parsing, ctx, state)?;
        }
        self.lz_tree.spa_for_symbol(state, params, sym)
    }

    fn spa(
        &self,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            let state = state.try_get_ensemble()?;
            if state.is_parallel() {
                return self.get_ensemble_spa_par(state, params, context_syms.unwrap_or(&[]));
            } else {
                return self.get_ensemble_spa(state, params, context_syms.unwrap_or(&[]));
            }
        }

        let state = state.try_get_lz78()?;
        if let Some(ctx) = context_syms {
            self.maybe_backshift_parse(params.backshift_parsing, ctx, state)?;
        }
        self.lz_tree.spa(state, params)
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let loss = -self
            .spa_for_symbol(sym, params, state, context_syms)?
            .log2();
        let params = params.try_get_lz78_mut()?;

        if params.ensemble != Ensemble::None {
            let states = state.try_get_ensemble()?;
            self.traverse_and_maybe_grow_ensemble(states, sym);
            return Ok(loss);
        }

        let state = state.try_get_lz78()?;
        self.lz_tree.traverse_one_symbol_frozen(state, sym);
        Ok(loss)
    }

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        let mut params = params.clone();
        Ok(Self {
            lz_tree: LZ78Tree::new(params.try_get_lz78_mut()?)?,
            n: 0,
            total_log_loss: 0.0,
        })
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl<S> GenerationSPA for LZ78SPA<S>
where
    S: GenerationSPATree,
{
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        if params.try_get_lz78()?.ensemble != Ensemble::None {
            let loss = -self.spa_for_symbol(sym, params, state, None)?.log2();
            let params = params.try_get_lz78_mut()?;
            let states = state.try_get_ensemble()?;
            if states.is_parallel() {
                states.pool.as_mut().unwrap().install(|| {
                    states.states.par_iter_mut().for_each(|state| {
                        self.lz_tree
                            .input_seed_data_symbol(state, sym, &mut params.clone())
                            .unwrap();
                    });
                })
            } else {
                for state in states.states.iter_mut() {
                    self.lz_tree.input_seed_data_symbol(state, sym, params)?;
                }
            }
            self.traverse_and_maybe_grow_ensemble(states, sym);
            return Ok(loss);
        }

        let params = params.try_get_lz78_mut()?;
        let state = state.try_get_lz78()?;
        let loss = self.lz_tree.input_seed_data_symbol(state, sym, params)?;
        self.lz_tree.traverse_one_symbol_frozen(state, sym);

        Ok(loss)
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
        let params = params.try_get_lz78_mut()?;
        if params.ensemble != Ensemble::None {
            let states = state.try_get_ensemble()?;
            let mut spa = self.get_ensemble_spa(states, params, context_syms)?;
            let (new_sym, sym_loss) = gen_symbol_from_spa(rng_sample, &mut spa, temperature, topk)?;
            self.traverse_and_maybe_grow_ensemble(states, new_sym);
            return Ok((new_sym, sym_loss));
        }

        let state = state.try_get_lz78()?;
        let bs_parse = params.backshift_parsing;
        self.maybe_backshift_parse(bs_parse, context_syms, state)?;

        let (new_sym, sym_loss) = self.lz_tree.generate_one_symbol(
            state,
            params,
            context_syms,
            rng_sample,
            temperature,
            topk,
        )?;
        self.lz_tree.traverse_one_symbol_frozen(state, new_sym);

        Ok((new_sym, sym_loss))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sequence::BinarySequence,
        spa::{
            dirichlet::DirichletSPATree,
            params::{DirichletParamsBuilder, LZ78ParamsBuilder},
        },
    };
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut params = LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum())
            .backshift(2, 1)
            .build_enum();
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&params).expect("failed to make LZ78SPA");
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
        let mut params = LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum())
            .backshift(2, 1)
            .build_enum();

        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&params).expect("failed to make LZ78SPA");
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

        let mut params = LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum())
            .backshift(2, 1)
            .ensemble(Ensemble::Entropy(1), true)
            .build_enum();

        let mut state = params.get_new_state();
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut params,
                &mut state,
            )
            .expect("failed to compute test loss");

        assert!((loss1 - loss2).abs() < 1e-4);

        let mut params = LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum())
            .backshift(2, 1)
            .ensemble(Ensemble::Depth(3), true)
            .build_enum();
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
    fn test_spa_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut params =
            LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum()).build_enum();
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa =
            LZ78SPA::<DirichletSPATree>::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.total_log_loss, new_spa.total_log_loss);
    }

    #[test]
    fn test_pruning() {
        let input = BinarySequence::from_data(bitvec![0, 1, 1, 0, 1, 1, 1].repeat(500));
        let mut params =
            LZ78ParamsBuilder::new(DirichletParamsBuilder::new(2).build_enum()).build_enum();
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let old_spas = spa.lz_tree.spa_tree.clone();

        let min_count = 5;
        spa.prune(min_count);

        println!(
            "Old num nodes: {}, new: {}",
            old_spas.ns.len(),
            spa.lz_tree.spa_tree.ns.len()
        );
        assert!(spa.lz_tree.spa_tree.ns.len() < old_spas.ns.len());

        // traverse the original and pruned trees to check
        let mut old_stack = vec![0u64];
        let mut new_stack = vec![0u64];
        while old_stack.len() > 0 {
            let old_node = old_stack.pop().unwrap();
            let old_count = old_spas.ns[old_node as usize];
            if old_count < min_count {
                continue;
            }

            assert!(new_stack.len() > 0);
            let new_node = new_stack.pop().unwrap();

            assert_eq!(old_count, spa.lz_tree.spa_tree.ns[new_node as usize]);

            old_stack.extend(
                [0, 1]
                    .iter()
                    .filter_map(|&x| old_spas.get_child_idx(old_node, x)),
            );

            new_stack.extend(
                [0, 1]
                    .iter()
                    .filter_map(|&x| spa.lz_tree.spa_tree.get_child_idx(new_node, x)),
            );
        }
    }
}
