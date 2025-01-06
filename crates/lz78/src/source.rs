use std::{collections::HashMap, sync::Arc};

use anyhow::{bail, Result};
use rand::{thread_rng, Rng};

use crate::{
    sequence::{Sequence, SequenceParams, U32Sequence},
    spa::{
        basic_spas::DirichletSPA,
        lz_transform::SPATree,
        states::{SPAState, LZ_ROOT_IDX},
        SPAParams, SPA,
    },
    util::sample_from_pdf,
};

/// Binary LZ78 proabaility source, where each node is associated with a
/// Bernoulli parameter, Theta. This node generates values i.i.d. Ber(Theta).
/// New child nodes draw Theta according to a discrete distribution defined
/// by `theta_pdf` and `theta_values`.
pub struct DiscreteBinaryThetaSPA {
    theta: f64,
    n: u64,
}

impl SPA for DiscreteBinaryThetaSPA {
    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        self.n += 1;
        Ok(-(self.spa_for_symbol(input, params, state)?.log2()))
    }

    fn spa_for_symbol(&self, sym: u32, _params: &SPAParams, _state: &mut SPAState) -> Result<f64> {
        match sym {
            0 => Ok(1.0 - self.theta),
            1 => Ok(self.theta),
            _ => bail!("DiscreteBinaryThetaSPA has an alphabet size of 2"),
        }
    }

    fn test_on_symbol(&self, input: u32, params: &SPAParams, state: &mut SPAState) -> Result<f64> {
        Ok(-(self.spa_for_symbol(input, params, state)?.log2()))
    }

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        if let SPAParams::DiscreteTheta(params) = params {
            let theta = params.theta_values
                [sample_from_pdf(&params.theta_pmf, thread_rng().gen_range(0.0..1.0)) as usize];
            Ok(Self { theta, n: 0 })
        } else {
            bail!("Expected params to be SPAParams::DiscreteTheta")
        }
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

pub trait SourceNodeSPA: SPA {
    fn new_with_rng(params: &SPAParams, rng: &mut impl Rng) -> Result<Self>
    where
        Self: Sized;
}

impl SourceNodeSPA for DirichletSPA {
    fn new_with_rng(params: &SPAParams, _rng: &mut impl Rng) -> Result<Self> {
        Self::new(params)
    }
}

impl SourceNodeSPA for DiscreteBinaryThetaSPA {
    fn new_with_rng(params: &SPAParams, rng: &mut impl Rng) -> Result<Self> {
        if let SPAParams::DiscreteTheta(params) = params {
            let theta = params.theta_values
                [sample_from_pdf(&params.theta_pmf, rng.gen_range(0.0..1.0)) as usize];
            Ok(Self { theta, n: 0 })
        } else {
            bail!("Expected params to be SPAParams::DiscreteTheta")
        }
    }
}

/// An LZ78-based probability source, which consists of an LZ78 prefix tree,
/// where each node has a corresponding SourceNode, which encapsulates how
/// values are generated from this probability source.
pub struct LZ78Source<S> {
    spa_tree: SPATree<S>,
    total_log_loss: f64,
    alphabet_size: u32,
    params: Arc<SPAParams>,
}

impl<S> LZ78Source<S>
where
    S: SourceNodeSPA,
{
    /// Given a SourceNode that is the root of the tree, creates an LZ78
    /// probability source
    pub fn new(params: &SPAParams, rng: &mut impl Rng) -> Result<Self> {
        let params = if let SPAParams::LZ78(x) = params {
            x.inner_params.clone()
        } else {
            bail!("Wrong params for building LZ78 SPA")
        };

        let alphabet_size = params.alphabet_size();
        let spa_tree = SPATree {
            spas: vec![S::new_with_rng(&params, rng)?],
            branch_mappings: vec![HashMap::new()],
            params: params.clone(),
        };

        Ok(Self {
            spa_tree,
            total_log_loss: 0.0,
            alphabet_size,
            params,
        })
    }

    /// Generates symbols from the probability source
    pub fn generate_symbols(
        &mut self,
        n: u64,
        rng: &mut impl Rng,
        state: &mut SPAState,
    ) -> Result<U32Sequence> {
        // output array
        let state = state.try_get_lz78()?;
        let mut syms = U32Sequence::new(&SequenceParams::AlphaSize(self.alphabet_size))?;

        for _ in 0..n {
            // generate the next symbol based on the PMF provided by the
            // current SourceNode
            let spa = self.spa_tree.spa(state)?;
            if spa.len() as u32 != self.alphabet_size {
                bail!("alphabet size specified incompatible with SourceNode implementation");
            }
            let next_sym = sample_from_pdf(&spa, rng.gen_range(0.0..1.0)) as u32;
            syms.put_sym(next_sym)?;

            self.total_log_loss += self.spa_tree.train_on_symbol(state, next_sym)?;

            let prev_node = state.node;
            self.spa_tree.traverse_one_symbol_frozen(state, next_sym);

            if state.node == LZ_ROOT_IDX {
                self.spa_tree.add_new_spa(
                    prev_node,
                    next_sym,
                    S::new_with_rng(self.params.as_ref(), rng)?,
                );
            }
        }

        Ok(syms)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_bernoulli_source() {
        let mut rng = thread_rng();
        let params = SPAParams::new_lz78(
            SPAParams::new_discrete(vec![0.5, 0.5], vec![0.0, 1.0]),
            false,
        );
        let mut state = params.get_new_state();
        let mut source: LZ78Source<DiscreteBinaryThetaSPA> =
            LZ78Source::new(&params, &mut rng).expect("failed to make source");

        let output = source
            .generate_symbols(100, &mut rng, &mut state)
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
        let params = SPAParams::new_lz78_dirichlet(4, 0.5, false);
        let mut state = params.get_new_state();
        let mut source: LZ78Source<DirichletSPA> =
            LZ78Source::new(&params, &mut rng).expect("failed to make source");

        let output = source
            .generate_symbols(50, &mut rng, &mut state)
            .expect("generation failed");

        println!("{:?}", output.data);
    }
}
