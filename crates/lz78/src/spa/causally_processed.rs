use std::{collections::HashMap, marker::PhantomData};

use super::{
    generation::{GenerationParams, GenerationSPA},
    lz_transform::{LZ78DebugState, LZ78SPA},
    states::SPAState,
    AdaptiveGamma, BackshiftParsing, Ensemble, SPAParams, SPA,
};
use crate::{
    sequence::{Sequence, SequenceParams},
    storage::ToFromBytes,
};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

pub struct CausalProcessedSequence<T1, T2> {
    pub original: T1,
    pub processed: T2,
}

impl<T1, T2> CausalProcessedSequence<T1, T2>
where
    T1: Sequence,
    T2: Sequence,
{
    pub fn new(original: T1, processed: T2) -> Result<Self> {
        if original.len() != processed.len() {
            bail!("Two sequences must be of the same length")
        }
        Ok(Self {
            original,
            processed,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.original.len()).map(|i| {
            (
                self.original.try_get(i).unwrap(),
                self.processed.try_get(i).unwrap(),
            )
        })
    }

    pub fn len(&self) -> u64 {
        self.original.len()
    }

    pub fn try_get(&self, i: u64) -> Result<(u32, u32)> {
        Ok((self.original.try_get(i)?, self.processed.try_get(i)?))
    }

    pub fn put_sym(&mut self, raw_sym: u32, processed_sym: u32) -> Result<()> {
        self.original.put_sym(raw_sym)?;
        self.processed.put_sym(processed_sym)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CausallyProcessedLZ78SPAParams {
    pub raw_data_lz78_params: SPAParams,
    pub processed_alpha_size: u32,
    pub raw_alpha_size: u32,
}

impl CausallyProcessedLZ78SPAParams {
    pub fn new(
        raw_alpha_size: u32,
        processed_alpha_size: u32,
        inner_params: SPAParams,
        default_gamma: Option<f64>,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        let raw_data_lz78_params = SPAParams::new_lz78(
            inner_params,
            default_gamma,
            adaptive_gamma,
            ensemble,
            backshift_parsing,
            debug,
        );

        Self {
            raw_data_lz78_params,
            processed_alpha_size,
            raw_alpha_size,
        }
    }

    pub fn new_dirichlet(
        raw_alpha_size: u32,
        processed_alpha_size: u32,
        gamma: f64,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        let raw_data_lz78_params = SPAParams::new_lz78_dirichlet(
            raw_alpha_size,
            gamma,
            adaptive_gamma,
            ensemble,
            backshift_parsing,
            debug,
        );
        Self {
            raw_data_lz78_params,
            processed_alpha_size,
            raw_alpha_size,
        }
    }

    pub fn get_new_state(&self) -> SPAState {
        SPAState::get_new_state(&self.raw_data_lz78_params)
    }
}

impl ToFromBytes for CausallyProcessedLZ78SPAParams {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend(self.raw_data_lz78_params.to_bytes()?);
        bytes.put_u32_le(self.processed_alpha_size);
        bytes.put_u32_le(self.raw_alpha_size);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let raw_data_lz78_params = SPAParams::from_bytes(bytes)?;
        let processed_alpha_size = bytes.get_u32_le();
        let raw_alpha_size = bytes.get_u32_le();
        Ok(Self {
            raw_data_lz78_params,
            processed_alpha_size,
            raw_alpha_size,
        })
    }
}

/// This is essentially identical to the LZ78 Transform SPA, but it traverses
/// the tree with a causally-processed version of the input rather than the
/// raw input symbols.
pub struct CausallyProcessedLZ78SPA<S> {
    lz78_spa: LZ78SPA<S>,
}

impl<S> CausallyProcessedLZ78SPA<S>
where
    S: SPA,
{
    pub fn train_on_block<T1, T2>(
        &mut self,
        input: &CausalProcessedSequence<T1, T2>,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
    ) -> Result<f64>
    where
        T1: Sequence,
        T2: Sequence,
    {
        let mut loss = 0.0;
        for (raw_sym, processed_sym) in input.iter() {
            loss += self.train_on_symbol(raw_sym, processed_sym, params, state)?;
        }
        Ok(loss)
    }

    pub fn train_on_symbol(
        &mut self,
        raw_input: u32,
        processed_input: u32,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;
        let loss = self.lz78_spa.update_current_node_spa(
            raw_input,
            state,
            params.raw_data_lz78_params.try_get_lz78_mut()?,
        )?;
        self.lz78_spa.update_tree_structure(
            processed_input,
            &mut params.raw_data_lz78_params,
            state,
        )?;

        Ok(loss)
    }

    pub fn spa(
        &self,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Vec<f64>> {
        self.lz78_spa
            .spa(&mut params.raw_data_lz78_params, state, context_syms)
    }

    pub fn spa_for_symbol(
        &self,
        raw_sym: u32,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        self.lz78_spa.spa_for_symbol(
            raw_sym,
            &mut params.raw_data_lz78_params,
            state,
            context_syms,
        )
    }

    // TODO: add ensemble
    pub fn test_on_block<T1, T2>(
        &self,
        input: &CausalProcessedSequence<T1, T2>,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
    ) -> Result<f64>
    where
        T1: Sequence,
        T2: Sequence,
    {
        let mut loss = 0.0;
        for (raw_sym, processed_sym) in input.iter() {
            loss += self.test_on_symbol(raw_sym, processed_sym, params, state)?;
        }
        Ok(loss)
    }

    // TODO: add ensemble
    pub fn test_on_symbol(
        &self,
        raw_input: u32,
        processed_input: u32,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;

        let loss = self.lz78_spa.spa_tree.test_on_symbol(
            state,
            params.raw_data_lz78_params.try_get_lz78_mut()?,
            raw_input,
        )?;
        self.lz78_spa
            .spa_tree
            .traverse_one_symbol_frozen(state, processed_input);

        Ok(loss)
    }

    pub fn new(params: &CausallyProcessedLZ78SPAParams) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            lz78_spa: LZ78SPA::<S>::new(&params.raw_data_lz78_params)?,
        })
    }

    pub fn num_symbols_seen(&self) -> u64 {
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
    pub fn input_seed_data_symbol(
        &self,
        raw_sym: u32,
        processed_sym: u32,
        params: &mut CausallyProcessedLZ78SPAParams,
        state: &mut SPAState,
    ) -> Result<f64> {
        let state = state.try_get_lz78()?;

        let loss = self.lz78_spa.spa_tree.input_seed_data_symbol(
            state,
            raw_sym,
            params.raw_data_lz78_params.try_get_lz78_mut()?,
        )?;

        self.lz78_spa
            .spa_tree
            .traverse_one_symbol_frozen(state, processed_sym);

        Ok(loss)
    }

    /// Returns a tuple of (new_raw_syum, processed_sym, log loss)
    pub fn generate_one_symbol<P>(
        &self,
        rng_sample: f64,
        processor: &P,
        raw_seq_history: &P::Input,
        params: &mut CausallyProcessedLZ78SPAParams,
        gen_params: &GenerationParams,
        state: &mut SPAState,
        context_syms: &[u32],
    ) -> Result<(u32, u32, f64)>
    where
        P: CausalProcessor,
    {
        let state = state.try_get_lz78()?;
        let params = params.raw_data_lz78_params.try_get_lz78_mut()?;
        self.lz78_spa
            .maybe_backshift_parse(params.backshift_parsing, context_syms, state)?;

        let (new_sym, sym_loss) = self.lz78_spa.spa_tree.generate_one_symbol(
            state,
            params,
            gen_params,
            context_syms,
            rng_sample,
        )?;
        let processed_sym = processor.process_symbol(new_sym, Some(raw_seq_history))?;
        self.lz78_spa
            .spa_tree
            .traverse_one_symbol_frozen(state, processed_sym);
        Ok((new_sym, processed_sym, sym_loss))
    }
}

pub fn generate_sequence_causally_processed<S, P>(
    spa: &mut CausallyProcessedLZ78SPA<S>,
    n: u64,
    spa_params: &mut CausallyProcessedLZ78SPAParams,
    gen_params: &GenerationParams,
    processor: &P,
    seed_data: Option<&CausalProcessedSequence<P::Input, P::Output>>,
    output_sequence: &mut CausalProcessedSequence<P::Input, P::Output>,
) -> Result<f64>
where
    P: CausalProcessor,
    S: GenerationSPA,
{
    let mut loss = 0.0;
    let mut state = spa_params.get_new_state();
    let mut output_syms = Vec::with_capacity(n as usize);

    if let Some(data) = seed_data {
        output_syms.reserve_exact(data.len() as usize);
        for (raw_sym, proc_sym) in data.iter() {
            loss += spa.input_seed_data_symbol(raw_sym, proc_sym, spa_params, &mut state)?;
            output_syms.push(proc_sym);
        }
    }

    let rng_samples = Uniform::new(0.0, 1.0)
        .sample_iter(&mut thread_rng())
        .take(n as usize)
        .collect_vec();

    for i in 0..n {
        let (raw_sym, proc_sym, new_loss) = spa.generate_one_symbol(
            rng_samples[i as usize],
            processor,
            &output_sequence.original,
            spa_params,
            gen_params,
            &mut state,
            &output_syms,
        )?;
        output_sequence.put_sym(raw_sym, proc_sym)?;
        output_syms.push(proc_sym);
        loss += new_loss;
    }
    Ok(loss)
}

pub trait CausalProcessor {
    type Input: Sequence;
    type Output: Sequence;

    fn process_sequence(&self, input: &Self::Input) -> Result<Self::Output>;

    fn process_symbol(&self, sym: u32, past: Option<&Self::Input>) -> Result<u32>;

    fn get_causally_processed_seq(
        &self,
        input: Self::Input,
    ) -> Result<CausalProcessedSequence<Self::Input, Self::Output>> {
        let output = self.process_sequence(&input)?;
        CausalProcessedSequence::new(input, output)
    }

    fn alphabet_size(&self) -> u32;
}

pub struct IntegerScalarQuantizer<T> {
    raw_alpha_size: u32,
    scale: u32,
    phantom: PhantomData<T>,
}

impl<T> IntegerScalarQuantizer<T> {
    pub fn new(raw_alpha_size: u32, scale: u32) -> Self {
        Self {
            raw_alpha_size,
            scale,
            phantom: PhantomData,
        }
    }
}

impl<T> CausalProcessor for IntegerScalarQuantizer<T>
where
    T: Sequence,
{
    type Input = T;
    type Output = T;

    fn process_sequence(&self, input: &Self::Input) -> Result<T> {
        let mut output = T::new(&SequenceParams::AlphaSize(self.alphabet_size()))?;
        for sym in input.iter() {
            output.put_sym(self.process_symbol(sym, None)?)?;
        }
        Ok(output)
    }

    fn process_symbol(&self, sym: u32, _past: Option<&Self::Input>) -> Result<u32> {
        Ok(sym.min(self.raw_alpha_size - 1) / self.scale)
    }

    fn alphabet_size(&self) -> u32 {
        self.raw_alpha_size / self.scale
    }
}

impl<T> ToFromBytes for IntegerScalarQuantizer<T> {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.raw_alpha_size);
        bytes.put_u32_le(self.scale);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let dynamic_range = bytes.get_u32_le();
        let scale = bytes.get_u32_le();
        Ok(Self::new(dynamic_range, scale))
    }
}

pub struct ManualQuantizer<T> {
    pub orig_params: SequenceParams,
    pub quant_params: SequenceParams,
    pub orig_to_quant_mapping: HashMap<u32, u32>,
    phantom_data: PhantomData<T>,
}

impl<T> ManualQuantizer<T> {
    pub fn new(
        orig_params: SequenceParams,
        quant_params: SequenceParams,
        orig_to_quant_mapping: HashMap<u32, u32>,
    ) -> Self {
        Self {
            orig_params,
            quant_params,
            orig_to_quant_mapping,
            phantom_data: PhantomData,
        }
    }
}

impl<T> CausalProcessor for ManualQuantizer<T>
where
    T: Sequence,
{
    type Input = T;
    type Output = T;

    fn process_sequence(&self, input: &Self::Input) -> Result<T> {
        let mut output = T::new(&self.quant_params)?;
        for sym in input.iter() {
            output.put_sym(self.process_symbol(sym, None)?)?;
        }

        Ok(output)
    }

    fn process_symbol(&self, sym: u32, _past: Option<&Self::Input>) -> Result<u32> {
        Ok(self.orig_to_quant_mapping[&sym])
    }

    fn alphabet_size(&self) -> u32 {
        self.quant_params.alphabet_size()
    }
}

impl<T> ToFromBytes for ManualQuantizer<T> {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend(self.orig_params.to_bytes()?);
        bytes.extend(self.quant_params.to_bytes()?);
        for i in 0..self.orig_params.alphabet_size() {
            bytes.put_u32_le(self.orig_to_quant_mapping[&i]);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let orig_params = SequenceParams::from_bytes(bytes)?;
        let quant_params = SequenceParams::from_bytes(bytes)?;
        let mut orig_to_quant_mapping: HashMap<u32, u32> = HashMap::new();

        for i in 0..orig_params.alphabet_size() {
            orig_to_quant_mapping.insert(i, bytes.get_u32_le());
        }

        Ok(Self {
            orig_params,
            quant_params,
            orig_to_quant_mapping,
            phantom_data: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{sequence::U8Sequence, spa::basic_spas::DirichletSPA};

    use super::*;

    #[test]
    fn test_scalar_quantizer() {
        let seq = U8Sequence::from_data(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 256).unwrap();
        let processor: IntegerScalarQuantizer<U8Sequence> = IntegerScalarQuantizer::new(256, 2);
        let out = processor.process_sequence(&seq).unwrap();
        assert_eq!(out.data, vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5]);
        assert_eq!(processor.process_symbol(32, Some(&seq)).unwrap(), 16);
    }

    #[test]
    fn sanity_check_log_loss() {
        let input = U8Sequence::from_data(vec![16, 32].repeat(500), 48).unwrap();
        let mut params = CausallyProcessedLZ78SPAParams::new_dirichlet(
            32,
            2,
            0.1,
            AdaptiveGamma::None,
            Ensemble::None,
            BackshiftParsing::Disabled,
            false,
        );
        let mut state = params.get_new_state();
        let mut spa: CausallyProcessedLZ78SPA<DirichletSPA> =
            CausallyProcessedLZ78SPA::new(&params).expect("failed to make LZ78SPA");

        let processor: IntegerScalarQuantizer<U8Sequence> = IntegerScalarQuantizer::new(48, 16);
        let processed_seq = processor
            .get_causally_processed_seq(input)
            .expect("could not get causally-processed sequence");
        spa.train_on_block(&processed_seq, &mut params, &mut state)
            .expect("failed to train spa");

        let test_seq_1 = processor
            .get_causally_processed_seq(
                U8Sequence::from_data(vec![16, 32, 16, 32, 16, 32, 16, 32, 16, 32, 16, 32], 48)
                    .unwrap(),
            )
            .expect("could not get causally-processed sequence");
        let loss1 = spa
            .test_on_block(&test_seq_1, &mut params, &mut state)
            .expect("failed to compute test loss");

        let test_seq_2 = processor
            .get_causally_processed_seq(
                U8Sequence::from_data(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 48).unwrap(),
            )
            .expect("could not get causally-processed sequence");
        let loss2 = spa
            .test_on_block(&test_seq_2, &mut params, &mut state)
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_tree_structure() {
        let input = U8Sequence::from_data(
            vec![0, 2, 0, 2, 0, 2, 3, 4, 6, 9, 2, 3, 4, 9, 9, 1, 2, 3, 4, 5].repeat(10),
            10,
        )
        .unwrap();
        let processor: IntegerScalarQuantizer<U8Sequence> = IntegerScalarQuantizer::new(10, 2);
        let processed_seq = processor
            .get_causally_processed_seq(input)
            .expect("could not get causally-processed sequence");

        let mut proc_params = CausallyProcessedLZ78SPAParams::new_dirichlet(
            10,
            2,
            0.1,
            AdaptiveGamma::None,
            Ensemble::None,
            BackshiftParsing::Disabled,
            false,
        );
        let mut proc_state = proc_params.get_new_state();
        let mut proc_spa: CausallyProcessedLZ78SPA<DirichletSPA> =
            CausallyProcessedLZ78SPA::new(&proc_params).expect("failed to make LZ78SPA");
        proc_spa
            .train_on_block(&processed_seq, &mut proc_params, &mut proc_state)
            .expect("failed to train spa");

        let mut params = SPAParams::new_lz78_dirichlet(
            5,
            0.1,
            AdaptiveGamma::None,
            Ensemble::None,
            BackshiftParsing::Disabled,
            false,
        );
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params).expect("failed to make LZ78SPA");
        spa.train_on_block(&processed_seq.processed, &mut params, &mut state)
            .expect("failed to train spa");

        assert_eq!(
            spa.spa_tree.branch_mappings,
            proc_spa.lz78_spa.spa_tree.branch_mappings
        );
    }
}
