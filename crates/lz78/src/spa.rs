use std::collections::{HashMap, HashSet};

use crate::{
    sequence::{Sequence, SequenceSlice, U32Sequence},
    storage::ToFromBytes,
    tree::LZ78Tree,
    util::sample_from_pdf,
};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng, Rng};

/// Interface for sequential probability assignments on data
pub trait SPA {
    /// Use a block of data to update the SPA. If `include_prev_context` is
    /// true, then this block is considered to be from the same sequence as
    /// the previous. Otherwise, it is assumed to be a separate sequence (e.g.,
    /// for the LZ78 SPA, this means we start at the root of the tree).
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence;

    /// Given a fixed SPA, compute the log loss of that SPA on a test sequence
    fn compute_test_loss<T: ?Sized>(
        &mut self,
        input: &T,
        include_prev_context: bool,
    ) -> Result<f64>
    where
        T: Sequence;

    /// Computes the SPA for one symbol in the alphabet
    fn compute_spa_for_sym_at_current_state(&mut self, sym: u32) -> f64;

    /// Computes the SPA for every symbol in the alphabet
    fn compute_spa_at_current_state(&mut self) -> Vec<f64>;

    /// Returns the normaliized log loss from training the SPA
    fn get_normalized_log_loss(&self) -> f64;

    /// Generates a sequence of data, using temperature and top-k sampling.
    /// For SPAs with the notion of a variable-length context, the `min_context`
    /// parameter specifies that the SPA tries to maintain a context length
    /// of at least a certain length.
    ///
    /// Using `seed_data`, you can specify that the sequence of generated data
    /// be the continuation of the specified sequence.
    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
        rng_samples: Option<&[f64]>,
    ) -> Result<f64>
    where
        T: Sequence;

    /// Makes a new SPA instance with the same parameters as `self`. Used for
    /// applying the LZ Transform to SPAs (which requires one SPA per layer of
    /// the tree)
    fn new_spa_with_same_params(&self) -> Self;
}

/// Functionality that specifically pertains to SPAs to which the LZ
/// transform is applied, i.e., SPAs that will be held at nodes of
/// an LZ78 tree. These are mainly for the purpose of allowing for
/// text generation.
pub trait SubSPA: SPA + Clone {
    fn train_on_symbol(&mut self, sym: u32) -> Result<f64>;

    fn compute_test_loss_on_symbol(&mut self, sym: u32) -> Result<f64>;

    fn reset_state(&mut self);

    /// Called at the beginning of text generation.
    fn prepare_for_generation(&mut self);

    /// Called at the end of sequence generation.
    fn cleanup_post_generation(&mut self);

    /// Called when "seeding" the text generation
    fn input_seed_data_symbol(&mut self, sym: u32) -> Result<f64>;

    /// Generates one symbol and updates the state of the SPA accordingly.
    fn generate_one_symbol(
        &mut self,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        rng_sample: f64,
    ) -> Result<(u32, f64)>;
}

#[derive(Debug, Clone)]
pub struct DirichletSPA {
    gamma: f64,
    alphabet_size: u32,
    counts: HashMap<u32, u64>,
    loss: f64,
    n: u64,
}

impl DirichletSPA {
    fn new(gamma: f64, alphabet_size: u32) -> Self {
        Self {
            gamma,
            alphabet_size,
            counts: HashMap::new(),
            loss: 0.0,
            n: 0,
        }
    }
}

impl ToFromBytes for DirichletSPA {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_f64_le(self.gamma);
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_u32_le(self.counts.len() as u32);
        for (&sym, &count) in self.counts.iter() {
            bytes.put_u32_le(sym);
            bytes.put_u64_le(count);
        }
        bytes.put_f64_le(self.loss);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let gamma = bytes.get_f64_le();
        let alphabet_size = bytes.get_u32_le();
        let counts_len = bytes.get_u32_le();

        let mut counts: HashMap<u32, u64> = HashMap::new();
        let mut n = 0;
        for _ in 0..counts_len {
            let sym = bytes.get_u32_le();
            let count = bytes.get_u64_le();
            n += count;
            counts.insert(sym, count);
        }
        let loss = bytes.get_f64_le();

        Ok(Self {
            gamma,
            alphabet_size,
            counts,
            loss,
            n,
        })
    }
}

impl SPA for DirichletSPA {
    fn train_on_block<T: ?Sized>(&mut self, input: &T, _include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for i in 0..input.len() {
            loss += self.train_on_symbol(input.try_get(i)?)?;
        }

        Ok(loss)
    }

    fn compute_test_loss<T: ?Sized>(
        &mut self,
        input: &T,
        _include_prev_context: bool,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for i in 0..input.len() {
            loss += self.compute_test_loss_on_symbol(input.try_get(i)?)?;
        }
        Ok(loss)
    }

    fn compute_spa_at_current_state(&mut self) -> Vec<f64> {
        (0..self.alphabet_size)
            .map(|x| self.compute_spa_for_sym_at_current_state(x))
            .collect_vec()
    }

    fn compute_spa_for_sym_at_current_state(&mut self, sym: u32) -> f64 {
        let sym_count = *self.counts.get(&sym).unwrap_or(&0) as f64;
        (sym_count + self.gamma) / (self.n as f64 + self.gamma * self.alphabet_size as f64)
    }

    fn get_normalized_log_loss(&self) -> f64 {
        self.loss / self.n as f64
    }

    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        _seed_data: Option<&T>,
        rng_samples: Option<&[f64]>,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sample_num in 0..len {
            let (new_sym, sym_loss) = self.generate_one_symbol(
                min_context,
                temperature,
                top_k,
                match rng_samples {
                    Some(x) => x[sample_num as usize],
                    None => thread_rng().gen::<f64>(),
                },
            )?;
            output_seq.put_sym(new_sym)?;
            loss += sym_loss;
        }
        Ok(loss)
    }

    fn new_spa_with_same_params(&self) -> Self {
        Self::new(self.gamma, self.alphabet_size)
    }
}

/// This SPA has no state, so most SubSPA functions are no-ops.
impl SubSPA for DirichletSPA {
    fn prepare_for_generation(&mut self) {}

    fn input_seed_data_symbol(&mut self, sym: u32) -> Result<f64> {
        Ok(self.compute_spa_for_sym_at_current_state(sym).log2())
    }

    fn generate_one_symbol(
        &mut self,
        _min_context: u64,
        temperature: f64,
        top_k: u32,
        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        // Compute the probability, according to the LZ78 SPA, that the
        // next symbol is x, for every x in the alphabet
        let mut spa = self.compute_spa_at_current_state();
        let most_likely_next_sym = (0..self.alphabet_size)
            .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
            .unwrap();

        // if temperature is 0.0, we just compute the argmax of the SPA. If
        // temperature is 1.0, the symbols are generated directly from the
        // SPA. In either case, we do not need the following computation.
        if temperature != 0.0 && temperature != 1.0 {
            spa = spa
                .iter()
                .map(|x| 2.0_f64.powf(x.log2() / temperature))
                .collect_vec();
        }

        // top-k sampling
        (0..self.alphabet_size)
            .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
            .take((self.alphabet_size - top_k) as usize)
            .map(|i| {
                spa[i as usize] = 0.0;
            })
            .collect_vec();

        let sum: f64 = spa.iter().sum();
        spa = spa.iter().map(|x| *x / sum).collect_vec();

        let new_sym = if temperature == 0.0 {
            most_likely_next_sym
        } else {
            sample_from_pdf(&spa, rng_sample) as u32
        };
        let loss = -self.compute_spa_for_sym_at_current_state(new_sym).log2();
        Ok((new_sym, loss))
    }

    fn cleanup_post_generation(&mut self) {}

    fn train_on_symbol(&mut self, sym: u32) -> Result<f64> {
        let loss = -self.compute_spa_for_sym_at_current_state(sym).log2();
        self.counts
            .insert(sym, self.counts.get(&sym).unwrap_or(&0) + 1);
        self.n += 1;
        Ok(loss)
    }

    fn compute_test_loss_on_symbol(&mut self, sym: u32) -> Result<f64> {
        Ok(-self.compute_spa_for_sym_at_current_state(sym).log2())
    }

    /// A DirichletSPA has no state, so this is a no-op.
    fn reset_state(&mut self) {}
}

#[derive(Debug, Clone)]
/// Currently stores a list of SPA nodes, and stores whether each node needs
/// to have its state reset.
struct LZTransformedNodes<S> {
    spa_nodes: Vec<S>,
    nodes_pending_reset: HashSet<u64>,
}

impl<S> LZTransformedNodes<S>
where
    S: SubSPA,
{
    fn new(root_node: S) -> Self {
        Self {
            spa_nodes: vec![root_node],
            nodes_pending_reset: HashSet::new(),
        }
    }

    fn reset_state(&mut self) {
        self.nodes_pending_reset = HashSet::<u64>::from_iter(0..self.spa_nodes.len() as u64);
    }

    // fn drain_reset_queue(&mut self) {
    //     for &node in self.nodes_pending_reset.iter() {
    //         self.spa_nodes[node as usize].reset_state();
    //     }
    //     self.nodes_pending_reset.clear();
    // }

    fn get_mut_ref(&mut self, i: u64) -> &mut S {
        if self.nodes_pending_reset.contains(&i) {
            self.spa_nodes[i as usize].reset_state();
            self.nodes_pending_reset.remove(&i);
        }
        &mut self.spa_nodes[i as usize]
    }

    fn grow(&mut self) {
        self.spa_nodes
            .push(self.spa_nodes[0].new_spa_with_same_params());
    }
}

impl<S> ToFromBytes for LZTransformedNodes<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.spa_nodes.len() as u64);
        for node in self.spa_nodes.iter() {
            bytes.extend(node.to_bytes()?);
        }

        bytes.put_u64_le(self.nodes_pending_reset.len() as u64);
        for &node in self.nodes_pending_reset.iter() {
            bytes.put_u64_le(node);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let num_nodes = bytes.get_u64_le() as usize;
        let mut spa_nodes: Vec<S> = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            spa_nodes.push(S::from_bytes(bytes)?);
        }

        let num_nodes_needing_reset = bytes.get_u64_le() as usize;
        let mut nodes_pending_reset: HashSet<u64> = HashSet::new();

        for _ in 0..num_nodes_needing_reset {
            nodes_pending_reset.insert(bytes.get_u64_le());
        }

        Ok(Self {
            spa_nodes,
            nodes_pending_reset,
        })
    }
}

/// LZ78 implementation of the sequential probability assignment
#[derive(Debug, Clone)]
pub struct LZ78SPA<S> {
    tree: LZ78Tree,
    spa_nodes: LZTransformedNodes<S>,
    state: u64,
    n: u64,
    total_log_loss: f64,
    alphabet_size: u32,

    // The following properties are for the SubSPA trait
    cached_state: u64,
    nodes_seen_in_generation: HashSet<u64>,
    generated_syms: U32Sequence,
}

impl LZ78SPA<DirichletSPA> {
    pub fn new_dirichlet(alpha_size: u32, gamma: f64) -> Self {
        Self::new(alpha_size, DirichletSPA::new(gamma, alpha_size))
    }
}

impl<S> LZ78SPA<S>
where
    S: SubSPA,
{
    pub fn new(alpha_size: u32, root_spa: S) -> Self {
        Self {
            tree: LZ78Tree::new(alpha_size),
            state: LZ78Tree::ROOT_IDX,
            spa_nodes: LZTransformedNodes::new(root_spa),
            n: 0,
            total_log_loss: 0.0,
            alphabet_size: alpha_size,
            cached_state: 0,
            nodes_seen_in_generation: HashSet::new(),
            generated_syms: U32Sequence::new(alpha_size),
        }
    }

    /// Traverses the tree  using a provided slice of the input sequence, and
    /// returns a tuple of the new state and log loss
    fn compute_test_loss_on_slice_from_state<'a, T: Sequence + ?Sized>(
        &mut self,
        input: SequenceSlice<'a, T>,
        state: u64,
    ) -> Result<(u64, f64)> {
        let mut log_loss = 0.0;
        let mut state = state;

        for sym in input.iter() {
            log_loss += self
                .spa_nodes
                .get_mut_ref(state)
                .compute_test_loss_on_symbol(sym)?;
            state = self.tree.traverse_one_symbol(state, sym);
        }

        Ok((state, log_loss))
    }
}

impl<S> ToFromBytes for LZ78SPA<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.n);
        bytes.put_u64_le(self.state);
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_f64_le(self.total_log_loss);
        bytes.extend(self.tree.to_bytes()?);
        bytes.extend(self.spa_nodes.to_bytes()?);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self> {
        let n = bytes.get_u64_le();
        let state = bytes.get_u64_le();
        let alphabet_size = bytes.get_u32_le();
        let total_log_loss = bytes.get_f64_le();
        let tree = LZ78Tree::from_bytes(bytes)?;
        let spa_nodes = LZTransformedNodes::<S>::from_bytes(bytes)?;

        Ok(Self {
            n,
            state,
            alphabet_size,
            total_log_loss,
            tree,
            spa_nodes,
            cached_state: LZ78Tree::ROOT_IDX,
            nodes_seen_in_generation: HashSet::new(),
            generated_syms: U32Sequence::new(alphabet_size),
        })
    }
}

impl<'a, S> SPA for LZ78SPA<S>
where
    S: SubSPA,
{
    /// Same as the LZ78 encoding process, but: (1) we don't actually compute
    /// the encoded bits, and (2) we compute the log loss incurred over the
    /// course of this block. By default, the LZ78Tree keeps track of the
    /// number of times each node was visited, which is sufficient to compute
    /// the SPA
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        let prev_log_loss = self.total_log_loss;
        if !include_prev_context {
            // reset the state to the root of the tree
            self.state = LZ78Tree::ROOT_IDX;
            self.spa_nodes.reset_state();
        }

        for sym in input.iter() {
            self.train_on_symbol(sym)?;
        }
        Ok(self.total_log_loss - prev_log_loss)
    }

    /// Compute the loss of a test sequence on this SPA
    fn compute_test_loss<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        Ok(self
            .compute_test_loss_on_slice_from_state(
                SequenceSlice::new(input, 0, input.len()),
                if include_prev_context {
                    self.state
                } else {
                    LZ78Tree::ROOT_IDX
                },
            )?
            .1)
    }

    fn compute_spa_at_current_state(&mut self) -> Vec<f64> {
        self.spa_nodes
            .get_mut_ref(self.state)
            .compute_spa_at_current_state()
    }

    fn compute_spa_for_sym_at_current_state(&mut self, sym: u32) -> f64 {
        self.spa_nodes
            .get_mut_ref(self.state)
            .compute_spa_for_sym_at_current_state(sym)
    }

    fn get_normalized_log_loss(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_log_loss / (self.n as f64)
        }
    }

    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
        rng_samples: Option<&[f64]>,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut log_loss: f64 = 0.0;
        let top_k = top_k.clamp(1, self.alphabet_size);

        let mut rng = thread_rng();

        // sample from the RNG once at the beginning for efficiency.
        // If rng_samples is defined, then use those instead.
        let new_samples = Uniform::new(0.0, 1.0)
            .sample_iter(&mut rng)
            .take(if rng_samples.is_some() {
                0
            } else {
                len as usize
            })
            .collect_vec();
        let samples = match rng_samples {
            Some(s) => s,
            None => &new_samples,
        };

        self.prepare_for_generation();

        // traverse the tree using the seed data
        if let Some(data) = seed_data {
            for sym in data.iter() {
                log_loss += self.input_seed_data_symbol(sym)?;
            }
        }

        for sample_num in 0..len {
            let (new_sym, sym_loss) = self.generate_one_symbol(
                min_context,
                temperature,
                top_k,
                samples[sample_num as usize],
            )?;
            output_seq.put_sym(new_sym)?;
            log_loss += sym_loss;
        }

        self.cleanup_post_generation();

        Ok(log_loss)
    }

    fn new_spa_with_same_params(&self) -> Self {
        Self::new(self.alphabet_size, self.spa_nodes.spa_nodes[0].clone())
    }
}

impl<'a, S> SubSPA for LZ78SPA<S>
where
    S: SubSPA,
{
    fn prepare_for_generation(&mut self) {
        self.cached_state = self.state;
        self.nodes_seen_in_generation.clear();
        self.generated_syms = U32Sequence::new(self.alphabet_size);
    }

    fn cleanup_post_generation(&mut self) {
        for &node in self.nodes_seen_in_generation.iter() {
            self.spa_nodes.get_mut_ref(node).cleanup_post_generation();
        }

        self.state = self.cached_state;
        self.nodes_seen_in_generation.clear();
        self.generated_syms = U32Sequence::new(self.alphabet_size);
    }

    fn input_seed_data_symbol(&mut self, sym: u32) -> Result<f64> {
        if !self.nodes_seen_in_generation.contains(&self.state) {
            self.nodes_seen_in_generation.insert(self.state);
            self.spa_nodes
                .get_mut_ref(self.state)
                .prepare_for_generation();
        }

        let loss = self
            .spa_nodes
            .get_mut_ref(self.state)
            .input_seed_data_symbol(sym);
        self.generated_syms.put_sym(sym)?;

        let single_sym_slice =
            SequenceSlice::new(&self.generated_syms, self.generated_syms.len() - 1, 1);
        let traversal_output =
            self.tree
                .traverse_to_leaf_from(self.state, single_sym_slice, false)?;

        self.state = if traversal_output.phrase_prefix_len == 0 {
            LZ78Tree::ROOT_IDX
        } else {
            traversal_output.state_idx
        };

        loss
    }

    fn generate_one_symbol(
        &mut self,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        rng_sample: f64,
    ) -> Result<(u32, f64)> {
        // If we're at a place with no information (root or leaf), we need to
        // re-seed the SPA with some context
        if self.state == LZ78Tree::ROOT_IDX || self.tree.is_leaf(self.state) {
            // keep on trying to re-seed the SPA: first start at min_context
            // symbols from the end, and traverse the prefix tree. If we
            // reach a leaf at any point, try with min_context - 1 symbols,
            // and repeat until the traversal does not reach a leaf.
            for k in (0..=min_context.min(self.generated_syms.len())).rev() {
                self.state = if k == 0 {
                    // we completely failed to re-seed the SPA, so we give
                    // up and generate the next symbol from the root
                    LZ78Tree::ROOT_IDX
                } else {
                    let mut state = self.state;
                    for sym in
                        SequenceSlice::new(&self.generated_syms, self.generated_syms.len() - k, k)
                            .iter()
                    {
                        state = self.tree.traverse_one_symbol(state, sym);
                        if state == LZ78Tree::ROOT_IDX {
                            break;
                        }
                    }
                    state
                };

                // re-seeding was successful!
                if !self.tree.is_leaf(self.state) && self.state != LZ78Tree::ROOT_IDX {
                    break;
                }
            }
        }

        if !self.nodes_seen_in_generation.contains(&self.state) {
            self.nodes_seen_in_generation.insert(self.state);
            self.spa_nodes
                .get_mut_ref(self.state)
                .prepare_for_generation();
        }
        let (new_sym, sym_loss) = self.spa_nodes.get_mut_ref(self.state).generate_one_symbol(
            min_context,
            temperature,
            top_k,
            rng_sample,
        )?;
        self.generated_syms.put_sym(new_sym)?;
        let traversal_output = self.tree.traverse_to_leaf_from(
            self.state,
            SequenceSlice::new(&self.generated_syms, self.generated_syms.len() - 1, 1),
            false,
        )?;
        self.state = if traversal_output.phrase_prefix_len == 0 {
            LZ78Tree::ROOT_IDX
        } else {
            traversal_output.state_idx
        };

        Ok((new_sym, sym_loss))
    }

    fn train_on_symbol(&mut self, sym: u32) -> Result<f64> {
        let loss = self
            .spa_nodes
            .get_mut_ref(self.state)
            .train_on_symbol(sym)?;
        self.total_log_loss += loss;
        self.state = self
            .tree
            .traverse_one_symbol_and_maybe_grow(self.state, sym);
        if self.state == LZ78Tree::ROOT_IDX {
            self.spa_nodes.grow();
        }
        self.n += 1;
        Ok(loss)
    }

    fn compute_test_loss_on_symbol(&mut self, sym: u32) -> Result<f64> {
        self.spa_nodes
            .get_mut_ref(self.state)
            .compute_test_loss_on_symbol(sym)
    }

    fn reset_state(&mut self) {
        self.state = LZ78Tree::ROOT_IDX;
        self.spa_nodes.reset_state();
    }
}

enum SPAType {
    LZDirichlet(LZ78SPA<DirichletSPA>),
    LZLZDirichlet(LZ78SPA<LZ78SPA<DirichletSPA>>),
}

impl ToFromBytes for SPAType {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SPAType::LZDirichlet(lz78_spa) => {
                bytes.put_u8(0);
                bytes.extend(lz78_spa.to_bytes()?);
            }
            SPAType::LZLZDirichlet(lz78_spa) => {
                bytes.put_u8(1);
                bytes.extend(lz78_spa.to_bytes()?);
            }
        };
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            0 => Ok(Self::LZDirichlet(LZ78SPA::<DirichletSPA>::from_bytes(
                bytes,
            )?)),
            1 => Ok(Self::LZLZDirichlet(
                LZ78SPA::<LZ78SPA<DirichletSPA>>::from_bytes(bytes)?,
            )),
            _ => bail!("unrecognized SPA type"),
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, CharacterSequence, U8Sequence};
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn test_dirichlet_to_from_bytes() {
        let mut spa = DirichletSPA::new(0.1, 3);
        spa.train_on_block(
            &U8Sequence::from_data(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 2, 2, 2, 1], 3)
                .unwrap(),
            false,
        )
        .expect("drain dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPA::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.counts, new_spa.counts);
    }

    #[test]
    fn sanity_check_log_loss() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut spa = LZ78SPA::new_dirichlet(2, 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let loss1 = spa
            .compute_test_loss(
                &BinarySequence::from_data(bitvec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                true,
            )
            .expect("failed to compute test loss");
        let loss2 = spa
            .compute_test_loss(
                &BinarySequence::from_data(bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                true,
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_lz_transformed_nodes_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut spa = LZ78SPA::new_dirichlet(2, 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let nodes_bytes = spa.spa_nodes.to_bytes().expect("nodes to bytes failed");
        let mut bytes: Bytes = nodes_bytes.into();
        let nodes = LZTransformedNodes::<DirichletSPA>::from_bytes(&mut bytes)
            .expect("nodes from bytes failed");
        assert_eq!(nodes.spa_nodes.len(), spa.spa_nodes.spa_nodes.len());
    }

    #[test]
    fn test_spa_to_from_bytes() {
        let input = BinarySequence::from_data(bitvec![0, 1].repeat(500));
        let mut spa = LZ78SPA::new_dirichlet(2, 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = LZ78SPA::<DirichletSPA>::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.total_log_loss, new_spa.total_log_loss);
    }

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let mut spa = LZ78SPA::new_dirichlet(input.alphabet_size(), 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.0,
            10,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            None,
        )
        .expect("generating data failed");

        println!(
            "Temperature 0, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output2 = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output2,
            100,
            10,
            1.0,
            1,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            None,
        )
        .expect("generating data failed");

        println!(
            "Temperature 1, topk 1, seed \"hello\": {:?}",
            generation_output2.data
        );

        assert_eq!(generation_output.data, generation_output2.data);

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            2.0,
            5,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            None,
        )
        .expect("generating data failed");

        println!(
            "Temperature 2, topk 5, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.5,
            10,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            None,
        )
        .expect("generating data failed");

        println!(
            "Temperature 0.5, topk 10, seed \"hello\": {:?}",
            generation_output.data
        );
    }
}
