use std::marker::PhantomData;

use super::{
    generation::{GenerationParams, GenerationSPA},
    lz_transform::{LZ78DebugState, LZ78SPA},
    SPAParams, SPA,
};
use crate::{sequence::Sequence, storage::ToFromBytes};
use anyhow::{bail, Result};

pub struct CausalProcessedSequence<T1, T2> {
    pub original: T1,
    pub processed: T2,
}

impl<T1, T2> CausalProcessedSequence<T1, T2>
where
    T1: Sequence,
    T2: Sequence,
{
    fn new(original: T1, processed: T2) -> Result<Self> {
        if original.len() != processed.len() {
            bail!("Two sequences must be of the same length")
        }
        Ok(Self {
            original,
            processed,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.original.len()).map(|i| {
            (
                self.original.try_get(i).unwrap(),
                self.processed.try_get(i).unwrap(),
            )
        })
    }

    fn len(&self) -> u64 {
        self.original.len()
    }

    fn try_get(&self, i: u64) -> Result<(u32, u32)> {
        Ok((self.original.try_get(i)?, self.processed.try_get(i)?))
    }

    fn put_sym(&mut self, raw_sym: u32, processed_sym: u32) -> Result<()> {
        self.original.put_sym(raw_sym)?;
        self.processed.put_sym(processed_sym)?;
        Ok(())
    }
}

/// This is essentially identical to the LZ78 Transform SPA, but it traverses
/// the tree with a causally-processed version of the input rather than the
/// raw input symbols.
struct CausallyProcessedLZ78SPA<S> {
    lz78_spa: LZ78SPA<S>,
}

impl<S> CausallyProcessedLZ78SPA<S>
where
    S: SPA,
{
    fn train_on_block<T1, T2>(
        &mut self,
        input: &CausalProcessedSequence<T1, T2>,
        params: &SPAParams,
    ) -> Result<f64>
    where
        T1: Sequence,
        T2: Sequence,
    {
        let mut loss = 0.0;
        for (raw_sym, processed_sym) in input.iter() {
            loss += self.train_on_symbol(raw_sym, processed_sym, params)?;
        }
        Ok(loss)
    }

    fn train_on_symbol(
        &mut self,
        raw_input: u32,
        processed_input: u32,
        params: &SPAParams,
    ) -> Result<f64> {
        let loss = self.lz78_spa.update_current_node_spa(raw_input)?;
        self.lz78_spa
            .update_tree_structure(processed_input, params)?;

        Ok(loss)
    }

    fn spa(&mut self, params: &SPAParams) -> Result<Vec<f64>> {
        let mut spa = Vec::with_capacity(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa.push(self.spa_for_symbol(sym, params)?);
        }
        Ok(spa)
    }

    fn spa_for_symbol(&mut self, raw_sym: u32, params: &SPAParams) -> Result<f64> {
        self.lz78_spa.spa_for_symbol(raw_sym, params)
    }

    fn test_on_block<T1, T2>(
        &mut self,
        input: &CausalProcessedSequence<T1, T2>,
        params: &SPAParams,
    ) -> Result<f64>
    where
        T1: Sequence,
        T2: Sequence,
    {
        let mut loss = 0.0;
        for (raw_sym, processed_sym) in input.iter() {
            loss += self.test_on_symbol(raw_sym, processed_sym, params)?;
        }
        Ok(loss)
    }

    fn test_on_symbol(
        &mut self,
        raw_input: u32,
        processed_input: u32,
        _params: &SPAParams,
    ) -> Result<f64> {
        let loss = self
            .lz78_spa
            .spa_tree
            .test_on_symbol(self.lz78_spa.state, raw_input)?;
        self.lz78_spa.state = self
            .lz78_spa
            .spa_tree
            .traverse_one_symbol_frozen(self.lz78_spa.state, processed_input);

        Ok(loss)
    }

    fn new(params: &super::SPAParams) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            lz78_spa: LZ78SPA::<S>::new(params)?,
        })
    }

    fn reset_state(&mut self) {
        self.lz78_spa.reset_state();
    }

    fn num_symbols_seen(&self) -> u64 {
        self.lz78_spa.num_symbols_seen()
    }

    pub fn get_normalized_log_loss(&self) -> f64 {
        self.lz78_spa.get_normalized_log_loss()
    }

    pub fn get_debug_info(&self) -> &LZ78DebugState {
        self.lz78_spa.get_debug_info()
    }

    pub fn clear_debug_info(&mut self) {
        self.lz78_spa.clear_debug_info();
    }
}

impl<S> ToFromBytes for CausallyProcessedLZ78SPA<S>
where
    S: ToFromBytes,
{
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        self.lz78_spa.to_bytes()
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            lz78_spa: LZ78SPA::<S>::from_bytes(bytes)?,
        })
    }
}

impl<S> CausallyProcessedLZ78SPA<S>
where
    S: GenerationSPA,
{
    fn cleanup_post_generation(&mut self) {
        self.lz78_spa.cleanup_post_generation();
    }

    fn input_seed_data_symbol(
        &mut self,
        raw_sym: u32,
        processed_sym: u32,
        params: &SPAParams,
    ) -> Result<f64> {
        self.lz78_spa.update_gen_state_with_curr_node();

        let loss = self.lz78_spa.spa_tree.input_seed_data_symbol(
            self.lz78_spa.get_gen_tree_state(),
            raw_sym,
            params,
        )?;

        // The generation state is used for reseeding the tree, so we update
        // it with the processed symbol
        self.lz78_spa.update_gen_state_with_sym(processed_sym);

        self.lz78_spa.traverse_tree_generation(processed_sym);

        Ok(loss)
    }

    /// Returns a tuple of (new_raw_syum, processed_sym, log loss)
    fn generate_one_symbol<P>(
        &mut self,
        rng_sample: f64,
        processor: &P,
        raw_seq_history: &P::Input,
        _params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, u32, f64)>
    where
        P: CausalProcessor,
    {
        self.lz78_spa.maybe_reseed_tree(gen_params);
        self.lz78_spa.update_gen_state_with_curr_node();

        let (new_sym, sym_loss) = self.lz78_spa.spa_tree.generate_one_symbol(
            self.lz78_spa.get_gen_tree_state(),
            gen_params,
            rng_sample,
        )?;
        let processed_sym = processor.process_symbol(new_sym, Some(raw_seq_history))?;
        self.lz78_spa.update_gen_state_with_sym(processed_sym);
        self.lz78_spa.traverse_tree_generation(processed_sym);
        Ok((new_sym, processed_sym, sym_loss))
    }
}

trait CausalProcessor {
    type Input: Sequence;
    type Output: Sequence;

    fn process_sequence(&self, input: &Self::Input, output: &mut Self::Output) -> Result<()>;

    fn process_symbol(&self, sym: u32, past: Option<&Self::Input>) -> Result<u32>;

    fn get_causally_processed_seq(
        &self,
        input: Self::Input,
    ) -> Result<CausalProcessedSequence<Self::Input, Self::Output>> {
        let processed = self.process_sequence(&input)?;
        CausalProcessedSequence::new(input, processed)
    }
}

struct ScalarQuantizer<T> {
    dynamic_range: u32,
    scale: u32,
    phantom: PhantomData<T>,
}

impl<T> CausalProcessor for ScalarQuantizer<T>
where
    T: Sequence,
{
    type Input = T;
    type Output = T;

    fn process_sequence(&self, input: &Self::Input, output: Self::Output) -> Result<()> {
        let mut result = Sel
    }

    fn process_symbol(&self, sym: u32, past: Option<&Self::Input>) -> Result<u32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
