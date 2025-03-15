use crate::storage::ToFromBytes;

use super::{
    config::{BackshiftParsing, Ensemble, LZ78Config, SPAConfig},
    generation::{gen_symbol_from_spa, GenerationSPA, GenerationSPATree},
    states::{LZ78State, SPAState, LZ_ROOT_IDX},
    util::adaptive_gamma,
    InfOutOptions, InferenceOutput, SPATree, SPA,
};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis};

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
    pub fn new(config: &LZ78Config) -> Result<Self> {
        Ok(Self {
            spa_tree: S::new(&config.inner_config)?,
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
        config: &mut LZ78Config,
        sym: u32,
    ) -> Result<()> {
        let prev_node = state.node;
        self.traverse_one_symbol_frozen(state, sym);
        if state.node == LZ_ROOT_IDX {
            // add a new leaf
            self.spa_tree
                .add_new(&config.inner_config, prev_node, sym)?;
        }
        Ok(())
    }

    fn apply_adaptive_gamma(&self, state: &mut LZ78State, config: &mut LZ78Config) {
        if let Some(gamma) = config.inner_config.maybe_get_gamma() {
            config.inner_config.maybe_set_gamma(adaptive_gamma(
                gamma,
                config.adaptive_gamma,
                state.depth,
                self.spa_tree.num_symbols_seen(state.node) + 1,
            ));
        }
    }

    pub fn train_on_symbol(
        &mut self,
        state: &mut LZ78State,
        config: &mut LZ78Config,
        sym: u32,
    ) -> Result<f64> {
        let mut old_gamma = None;
        let node = state.node;

        if config.inner_config.compute_training_loss() {
            old_gamma = config.inner_config.maybe_get_gamma();
            self.apply_adaptive_gamma(state, config);
        }

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);
        let loss =
            self.spa_tree
                .train_spa_on_symbol(node, sym, &mut config.inner_config, inner_state)?;
        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn test_on_symbol(
        &self,
        state: &mut LZ78State,
        config: &mut LZ78Config,
        sym: u32,
    ) -> Result<f64> {
        let old_gamma = config.inner_config.maybe_get_gamma();
        let node = state.node;
        self.apply_adaptive_gamma(state, config);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);

        let loss =
            self.spa_tree
                .test_on_symbol(node, sym, &mut config.inner_config, inner_state, None)?;
        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn spa_for_symbol(
        &self,
        state: &mut LZ78State,
        config: &mut LZ78Config,
        sym: u32,
    ) -> Result<f64> {
        let node = state.node;
        let old_gamma = config.inner_config.maybe_get_gamma();
        self.apply_adaptive_gamma(state, config);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);

        let spa =
            self.spa_tree
                .spa_for_symbol(node, sym, &mut config.inner_config, inner_state, None)?;

        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
        }
        Ok(spa)
    }

    pub fn spa(&self, state: &mut LZ78State, config: &mut LZ78Config) -> Result<Array1<f64>> {
        let node = state.node;
        let old_gamma = config.inner_config.maybe_get_gamma();
        self.apply_adaptive_gamma(state, config);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);

        let spa = self
            .spa_tree
            .spa(node, &mut config.inner_config, inner_state, None)?;

        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
        }
        Ok(spa)
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
        config: &mut LZ78Config,
    ) -> Result<f64> {
        let node = state.node;
        let old_gamma = config.inner_config.maybe_get_gamma();
        self.apply_adaptive_gamma(state, config);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);
        let loss = self.spa_tree.input_seed_data_symbol(
            node,
            sym,
            &mut config.inner_config,
            inner_state,
        )?;

        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
        }
        Ok(loss)
    }

    pub fn generate_one_symbol(
        &self,
        state: &mut LZ78State,
        config: &mut LZ78Config,
        context_syms: &[u32],
        rng_sample: f64,
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)> {
        let node = state.node;

        let old_gamma = config.inner_config.maybe_get_gamma();
        self.apply_adaptive_gamma(state, config);

        let mut none_state = SPAState::None;
        let inner_state = state
            .get_child_state(&mut config.inner_config)
            .unwrap_or(&mut none_state);

        let res = self.spa_tree.generate_one_symbol(
            node,
            rng_sample,
            &mut config.inner_config,
            inner_state,
            context_syms,
            temperature,
            topk,
        )?;

        if let Some(gamma) = old_gamma {
            config.inner_config.maybe_set_gamma(gamma);
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
        config: &mut LZ78Config,
    ) -> Result<f64> {
        let loss = self.lz_tree.train_on_symbol(state, config, sym)?;
        self.total_log_loss += loss;
        Ok(loss)
    }

    pub fn update_tree_structure(
        &mut self,
        sym: u32,
        config: &mut LZ78Config,
        state: &mut LZ78State,
    ) -> Result<()> {
        self.lz_tree
            .traverse_one_symbol_and_maybe_grow(state, config, sym)?;
        self.n += 1;

        Ok(())
    }

    pub fn follow_prefix(&self, context: &[u32], state: &mut LZ78State) {
        state.go_to_root();
        for sym in context {
            self.lz_tree.traverse_one_symbol_frozen(state, *sym);
            if state.node == LZ_ROOT_IDX {
                return;
            }
        }
    }

    pub fn maybe_backshift_parse(
        &self,
        bs_parse_config: BackshiftParsing,
        reseeding_seq: &[u32],
        state: &mut LZ78State,
    ) -> Result<()> {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context
        let (desired_context_length, break_at_phrase) = if let BackshiftParsing::Enabled {
            desired_context_length,
            break_at_phrase,
        } = bs_parse_config
        {
            (desired_context_length, break_at_phrase)
        } else {
            return Ok(());
        };

        if self.lz_tree.spa_tree.num_symbols_seen(state.node) == 0 && state.store_patches {
            state.patch_information.push(
                ((state.internal_counter - state.depth as u64)..state.internal_counter)
                    .collect_vec(),
            );
        }

        if state.node == LZ_ROOT_IDX || self.lz_tree.spa_tree.num_symbols_seen(state.node) == 0 {
            let reseeding_start =
                reseeding_seq.len() - (reseeding_seq.len()).min(desired_context_length as usize);
            let reseeding_end = reseeding_seq.len();

            // keep on trying to re-seed the SPA
            for start_idx in reseeding_start..reseeding_end {
                state.go_to_root();

                for idx in start_idx..reseeding_end {
                    self.lz_tree
                        .traverse_one_symbol_frozen(state, reseeding_seq[idx]);
                    if state.node == LZ_ROOT_IDX && break_at_phrase {
                        break;
                    }
                }

                // re-seeding was successful!
                if state.node != LZ_ROOT_IDX
                    && self.lz_tree.spa_tree.num_symbols_seen(state.node) > 0
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

    fn get_ensemble_spa(
        &self,
        state: &mut LZ78State,
        config: &mut LZ78Config,
        context_syms: &[u32],
    ) -> Result<Array1<f64>> {
        self.maybe_backshift_parse(config.backshift_parsing, context_syms, state)?;

        let mut state_vec = vec![state.clone()];

        for depth_level in (4..state.depth as usize).rev() {
            if depth_level > context_syms.len() {
                break;
            }
            let mut new_state = state.clone();
            self.follow_prefix(
                &context_syms[context_syms.len() - depth_level..],
                &mut new_state,
            );
            if new_state.depth > 0 && self.lz_tree.spa_tree.num_symbols_seen(new_state.node) > 0 {
                state_vec.push(new_state);
            }
            if state_vec.len() == config.ensemble.get_num_states() {
                break;
            }
        }

        if state_vec.len() == 1 {
            return self.lz_tree.spa(state, config);
        }

        let mut spas = Array2::zeros((
            state_vec.len(),
            config.inner_config.alphabet_size() as usize,
        ));
        let mut depths: Array1<f64> = Array1::zeros(state_vec.len());

        for (i, state) in state_vec.iter_mut().enumerate() {
            let mut row = spas.index_axis_mut(Axis(0), i);
            row += &self.lz_tree.spa(state, &mut config.clone()).unwrap();
            self.lz_tree.spa(state, &mut config.clone()).unwrap();
            depths[i] = state.depth as f64;
        }

        self.ensemble_spa_from_spas(spas, depths, config.ensemble)
    }
}

impl<S> LZ78SPA<S>
where
    S: SPATree,
{
    pub fn shrink_to_fit(&mut self) {
        self.lz_tree.spa_tree.shrink_to_fit();
    }
}

impl<S> SPA for LZ78SPA<S>
where
    S: SPATree,
{
    fn train_on_symbol(
        &mut self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        let config = config.try_get_lz78_mut()?;
        let state = state.try_get_lz78()?;
        let loss = self.update_current_node_spa(sym, state, config)?;
        self.update_tree_structure(sym, config, state)?;

        Ok(loss)
    }

    fn spa_for_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let config = config.try_get_lz78_mut()?;
        if config.ensemble != Ensemble::None {
            return Ok(self.get_ensemble_spa(
                state.try_get_lz78()?,
                config,
                context_syms.unwrap_or(&[]),
            )?[sym as usize]);
        }

        let state = state.try_get_lz78()?;
        if let Some(ctx) = context_syms {
            self.maybe_backshift_parse(config.backshift_parsing, ctx, state)?;
        }
        self.lz_tree.spa_for_symbol(state, config, sym)
    }

    fn spa(
        &self,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let config = config.try_get_lz78_mut()?;
        if config.ensemble != Ensemble::None {
            return self.get_ensemble_spa(
                state.try_get_lz78()?,
                config,
                context_syms.unwrap_or(&[]),
            );
        }

        let state = state.try_get_lz78()?;
        if let Some(ctx) = context_syms {
            self.maybe_backshift_parse(config.backshift_parsing, ctx, state)?;
        }
        self.lz_tree.spa(state, config)
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
    ) -> Result<InferenceOutput> {
        let inf_out = if inf_out_options.output_probs() {
            let dist = self.spa(config, state, context_syms)?;
            let loss = -dist[sym as usize].log2();
            let ppl = loss.exp2();
            InferenceOutput::new(loss, ppl, vec![loss], vec![dist.to_vec()])
        } else {
            let loss = -self
                .spa_for_symbol(sym, config, state, context_syms)?
                .log2();
            let ppl = loss.exp2();
            let losses = if inf_out_options.output_losses() {
                vec![loss]
            } else {
                vec![]
            };

            InferenceOutput::new(loss, ppl, losses, vec![])
        };

        let state = state.try_get_lz78()?;
        let old_depth = state.depth;
        self.lz_tree.traverse_one_symbol_frozen(state, sym);
        // println!("{} {} {}", state.internal_counter, state.depth, old_depth);
        if state.node == LZ_ROOT_IDX && state.store_patches {
            state.patch_information.push(
                ((state.internal_counter - old_depth as u64)..state.internal_counter).collect_vec(),
            );
        }
        if state.store_patches {
            state.internal_counter += 1;
        }
        Ok(inf_out)
    }

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized,
    {
        let mut config = config.clone();
        Ok(Self {
            lz_tree: LZ78Tree::new(config.try_get_lz78_mut()?)?,
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
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64> {
        let config = config.try_get_lz78_mut()?;
        let state = state.try_get_lz78()?;
        let loss = self.lz_tree.input_seed_data_symbol(state, sym, config)?;
        self.lz_tree.traverse_one_symbol_frozen(state, sym);

        Ok(loss)
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
        let config = config.try_get_lz78_mut()?;
        if config.ensemble != Ensemble::None {
            let state = state.try_get_lz78()?;
            let mut spa = self.get_ensemble_spa(state, config, context_syms)?;
            let (new_sym, sym_loss) = gen_symbol_from_spa(rng_sample, &mut spa, temperature, topk)?;
            self.lz_tree.traverse_one_symbol_frozen(state, new_sym);
            return Ok((new_sym, sym_loss));
        }

        let state = state.try_get_lz78()?;
        let bs_parse = config.backshift_parsing;
        self.maybe_backshift_parse(bs_parse, context_syms, state)?;

        let (new_sym, sym_loss) = self.lz_tree.generate_one_symbol(
            state,
            config,
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
            config::{DirichletConfigBuilder, LZ78ConfigBuilder},
            dirichlet::DirichletSPATree,
        },
    };
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut config = LZ78ConfigBuilder::new(DirichletConfigBuilder::new(2).build_enum())
            .backshift(2, true)
            .build_enum();
        let mut state = config.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&config).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &mut config, &mut state)
            .expect("failed to train spa");

        state.reset();
        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        state.reset();
        let loss2 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn sanity_check_ensemble() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(1000));
        let mut config = LZ78ConfigBuilder::new(DirichletConfigBuilder::new(2).build_enum())
            .backshift(2, true)
            .build_enum();

        let mut state = config.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&config).expect("failed to make LZ78SPA");
        spa.train_on_block(&input, &mut config, &mut state)
            .expect("failed to train spa");

        state.reset();
        let loss1 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        let mut config = LZ78ConfigBuilder::new(DirichletConfigBuilder::new(2).build_enum())
            .backshift(2, true)
            .ensemble(Ensemble::Depth(3))
            .build_enum();
        let mut state = config.get_new_state();
        let loss3 = spa
            .test_on_block(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                &mut config,
                &mut state,
                InfOutOptions::Basic,
                None,
            )
            .expect("failed to compute test loss")
            .avg_log_loss;

        println!("without ensemble = {loss1}, with ensemble = {loss3}")
    }

    #[test]
    fn test_spa_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut config =
            LZ78ConfigBuilder::new(DirichletConfigBuilder::new(2).build_enum()).build_enum();
        let mut state = config.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&config).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut config, &mut state)
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
        let mut config =
            LZ78ConfigBuilder::new(DirichletConfigBuilder::new(2).build_enum()).build_enum();
        let mut state = config.get_new_state();
        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&config).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut config, &mut state)
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
