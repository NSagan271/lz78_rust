use anyhow::{bail, Result};
use hashbrown::{HashMap, HashSet};
use rand::{thread_rng, Rng};

use crate::{
    sequence::{Sequence, SequenceParams, U32Sequence},
    spa::{
        dirichlet::DirichletSPATree,
        lz_transform::LZ78Tree,
        params::SPAParams,
        states::{SPAState, LZ_ROOT_IDX},
        LZWTree, SPATree,
    },
    util::sample_from_pdf,
};

/// Binary LZ78 proabaility source, where each node is associated with a
/// Bernoulli parameter, Theta. This node generates values i.i.d. Ber(Theta).
/// New child nodes draw Theta according to a discrete distribution defined
/// by `theta_pdf` and `theta_values`.
pub struct DiscreteBinaryThetaSPATree {
    thetas: Vec<f64>,
    ns: Vec<u64>,
    branches: LZWTree,
}

impl SPATree for DiscreteBinaryThetaSPATree {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.ns[idx as usize] += 1;
        Ok(-(self.spa_for_symbol(idx, sym, params, state, None)?.log2()))
    }

    fn spa_for_symbol(
        &self,
        idx: u64,
        sym: u32,
        _params: &mut SPAParams,
        _state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        match sym {
            0 => Ok(1.0 - self.thetas[idx as usize]),
            1 => Ok(self.thetas[idx as usize]),
            _ => bail!("DiscreteBinaryThetaSPA has an alphabet size of 2"),
        }
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        input: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-(self
            .spa_for_symbol(idx, input, params, state, context_syms)?
            .log2()))
    }

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new_with_rng(params, &mut thread_rng())
    }

    fn num_symbols_seen(&self, idx: u64) -> u64 {
        self.ns[idx as usize]
    }

    fn add_new(&mut self, params: &SPAParams, parent_idx: u64, sym: u32) -> Result<()> {
        self.add_new_with_rng(params, parent_idx, sym, &mut thread_rng())
    }

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        self.branches.get_child_idx(idx, sym)
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
                self.thetas[write_idx] = self.thetas[i];
                replace.insert(i as u64, write_idx as u64);
                write_idx += 1;
            }
        }
        self.ns.truncate(write_idx.max(1));
        self.thetas.truncate(write_idx.max(1));
        self.branches.remove_batch(&remove);
        self.branches.replace(&replace);
    }

    fn shrink_to_fit(&mut self) {
        self.thetas.shrink_to_fit();
        self.ns.shrink_to_fit();
        self.branches.shrink_to_fit();
    }
}

pub struct DiracDirichletMixtureTree {
    is_dirichlet: Vec<bool>,
    thetas: Vec<f64>,
    dirichlet_spa: DirichletSPATree,
}

impl SPATree for DiracDirichletMixtureTree {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let loss = -self.spa_for_symbol(idx, sym, params, state, None)?.log2();
        if self.is_dirichlet[idx as usize] {
            self.dirichlet_spa
                .train_spa_on_symbol(idx, sym, params, state)?;
        }

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
        if self.is_dirichlet[idx as usize] {
            let params = params.try_get_dirac_mut()?;
            return self.dirichlet_spa.spa_for_symbol(
                idx,
                sym,
                &mut params.dirichlet_params,
                state,
                context_syms,
            );
        }
        match sym {
            0 => Ok(1.0 - self.thetas[idx as usize]),
            1 => Ok(self.thetas[idx as usize]),
            _ => bail!("DiracDirichletMixtureTree has an alphabet size of 2"),
        }
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-(self
            .spa_for_symbol(idx, sym, params, state, context_syms)?
            .log2()))
    }

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new_with_rng(params, &mut thread_rng())
    }

    fn num_symbols_seen(&self, idx: u64) -> u64 {
        self.dirichlet_spa.num_symbols_seen(idx)
    }

    fn add_new(&mut self, params: &SPAParams, parent_idx: u64, sym: u32) -> Result<()> {
        self.add_new_with_rng(params, parent_idx, sym, &mut thread_rng())
    }

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        self.dirichlet_spa.get_child_idx(idx, sym)
    }

    fn prune(&mut self, min_count: u64) {
        let mut write_idx = 0;
        for i in 0..self.dirichlet_spa.ns.len() {
            if self.dirichlet_spa.ns[i] >= min_count {
                self.thetas[write_idx] = self.thetas[i];
                self.is_dirichlet[write_idx] = self.is_dirichlet[i];
                write_idx += 1;
            }
        }
        self.dirichlet_spa.prune(min_count);
    }

    fn shrink_to_fit(&mut self) {
        self.dirichlet_spa.shrink_to_fit();
        self.thetas.shrink_to_fit();
        self.is_dirichlet.shrink_to_fit();
    }
}

pub trait SourceNodeSPATree: SPATree {
    fn add_new_with_rng(
        &mut self,
        params: &SPAParams,
        parent_idx: u64,
        sym: u32,
        rng: &mut impl Rng,
    ) -> Result<()>;

    fn new_with_rng(params: &SPAParams, rng: &mut impl Rng) -> Result<Self>
    where
        Self: Sized;
}

impl SourceNodeSPATree for DirichletSPATree {
    fn add_new_with_rng(
        &mut self,
        params: &SPAParams,
        parent_idx: u64,
        sym: u32,
        _rng: &mut impl Rng,
    ) -> Result<()> {
        self.add_new(params, parent_idx, sym)
    }

    fn new_with_rng(params: &SPAParams, _rng: &mut impl Rng) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new(params)
    }
}

impl SourceNodeSPATree for DiscreteBinaryThetaSPATree {
    fn new_with_rng(params: &SPAParams, rng: &mut impl Rng) -> Result<Self> {
        let params = params.try_get_discrete()?;
        let theta = params.theta_values
            [sample_from_pdf(&params.theta_pmf, rng.gen_range(0.0..1.0)) as usize];
        Ok(Self {
            thetas: vec![theta],
            ns: vec![0],
            branches: LZWTree::new(),
        })
    }

    fn add_new_with_rng(
        &mut self,
        params: &SPAParams,
        parent_idx: u64,
        sym: u32,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let params = params.try_get_discrete()?;
        let theta = params.theta_values
            [sample_from_pdf(&params.theta_pmf, rng.gen_range(0.0..1.0)) as usize];
        self.branches
            .add_leaf(parent_idx, sym, self.ns.len() as u64);
        self.ns.push(0);
        self.thetas.push(theta);

        Ok(())
    }
}

impl SourceNodeSPATree for DiracDirichletMixtureTree {
    fn new_with_rng(params: &SPAParams, rng: &mut impl Rng) -> Result<Self>
    where
        Self: Sized,
    {
        let params = params.try_get_dirac()?;
        let is_dirichlet = rng.gen_bool(params.dirichlet_weight);
        let theta = params.disc_params.theta_values
            [sample_from_pdf(&params.disc_params.theta_pmf, rng.gen_range(0.0..1.0)) as usize];
        return Ok(Self {
            is_dirichlet: vec![is_dirichlet],
            thetas: vec![theta],
            dirichlet_spa: DirichletSPATree::new(&params.dirichlet_params)?,
        });
    }

    fn add_new_with_rng(
        &mut self,
        params: &SPAParams,
        parent_idx: u64,
        sym: u32,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let params = params.try_get_dirac()?;
        let is_dirichlet = rng.gen_bool(params.dirichlet_weight);
        let theta = params.disc_params.theta_values
            [sample_from_pdf(&params.disc_params.theta_pmf, rng.gen_range(0.0..1.0)) as usize];
        self.is_dirichlet.push(is_dirichlet);
        self.thetas.push(theta);
        self.dirichlet_spa
            .add_new(&params.dirichlet_params, parent_idx, sym)
    }
}

/// An LZ78-based probability source, which consists of an LZ78 prefix tree,
/// where each node has a corresponding SourceNode, which encapsulates how
/// values are generated from this probability source.
pub struct LZ78Source<S> {
    pub spa_tree: LZ78Tree<S>,
    pub total_log_loss: f64,
    pub n: u64,
}

impl<S> LZ78Source<S>
where
    S: SourceNodeSPATree,
{
    /// Given a SourceNode that is the root of the tree, creates an LZ78
    /// probability source
    pub fn new(params: &SPAParams, rng: &mut impl Rng) -> Result<Self> {
        let params = params.try_get_lz78()?;
        Ok(Self {
            spa_tree: LZ78Tree {
                spa_tree: S::new_with_rng(&params.inner_params, rng)?,
            },
            total_log_loss: 0.0,
            n: 0,
        })
    }

    /// Generates symbols from the probability source
    pub fn generate_symbols(
        &mut self,
        n: u64,
        rng: &mut impl Rng,
        params: &mut SPAParams,
        state: &mut SPAState,
    ) -> Result<U32Sequence> {
        // output array
        let state = state.try_get_lz78()?;
        let mut syms = U32Sequence::new(&SequenceParams::AlphaSize(params.alphabet_size()))?;
        let mut lz_params = params.try_get_lz78_mut()?.clone();

        for _ in 0..n {
            // generate the next symbol based on the PMF provided by the
            // current SourceNode
            let spa = self.spa_tree.spa(state, &mut lz_params)?;
            let next_sym = sample_from_pdf(&spa, rng.gen_range(0.0..1.0)) as u32;
            syms.put_sym(next_sym)?;

            self.total_log_loss +=
                self.spa_tree
                    .train_on_symbol(state, &mut lz_params, next_sym)?;

            let prev_node = state.node;
            self.spa_tree.traverse_one_symbol_frozen(state, next_sym);
            self.n += 1;

            if state.node == LZ_ROOT_IDX {
                self.spa_tree
                    .spa_tree
                    .add_new_with_rng(params, prev_node, next_sym, rng)?;
            }
        }

        Ok(syms)
    }
}

#[cfg(test)]
mod tests {
    use crate::spa::params::{DirichletParamsBuilder, DiscreteThetaParams, LZ78ParamsBuilder};

    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_bernoulli_source() {
        let mut rng = thread_rng();
        let mut params = LZ78ParamsBuilder::new(SPAParams::Discrete(DiscreteThetaParams::new(
            &[0.5, 0.5],
            &[0.0, 1.0],
        )))
        .build_enum();
        let mut state = params.get_new_state();
        let mut source: LZ78Source<DiscreteBinaryThetaSPATree> =
            LZ78Source::new(&mut params, &mut rng).expect("failed to make source");

        let output = source
            .generate_symbols(100, &mut rng, &mut params, &mut state)
            .expect("generation failed");

        let mut i = 0;
        let mut phrase_num = 0;
        while i + 2 * phrase_num + 1 < output.len() {
            assert_eq!(
                output.data[i as usize..=(i + phrase_num) as usize],
                output.data[(i + phrase_num + 1) as usize..=(i + 2 * phrase_num + 1) as usize]
            );
            i += phrase_num + 1;
            phrase_num += 1;
        }
    }

    #[test]
    fn sanity_check_lz778_source() {
        let mut rng = thread_rng();
        let mut params = DirichletParamsBuilder::new(4).build_enum();
        let mut state = params.get_new_state();
        let mut source: LZ78Source<DirichletSPATree> =
            LZ78Source::new(&mut params, &mut rng).expect("failed to make source");

        let output = source
            .generate_symbols(50, &mut rng, &mut params, &mut state)
            .expect("generation failed");

        println!("{:?}", output.data);
    }
}
