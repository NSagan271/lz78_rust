use anyhow::Result;
use config::SPAConfig;
use ndarray::Array1;
use states::SPAState;

use crate::sequence::Sequence;

pub mod config;
pub mod dirichlet;
pub mod generation;
pub mod lz_transform;
pub mod lzw_tree;
pub mod ngram;
pub mod states;
pub mod util;

pub trait SPATree: Sync {
    fn train_spa_on_symbol(
        &mut self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
    ) -> Result<f64>;

    fn spa_for_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn spa(
        &self,
        idx: u64,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let mut spa = Array1::zeros(config.alphabet_size() as usize);
        for sym in 0..config.alphabet_size() {
            spa[sym as usize] = self.spa_for_symbol(idx, sym, config, state, context_syms)?;
        }
        Ok(spa)
    }

    fn test_on_symbol(
        &self,
        idx: u64,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn add_new(&mut self, config: &SPAConfig, parent_idx: u64, sym: u32) -> Result<()>;

    fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64>;

    fn num_symbols_seen(&self, idx: u64) -> u64;

    fn num_nodes(&self) -> u64;

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized;

    fn prune(&mut self, min_count: u64);

    fn shrink_to_fit(&mut self);
}

pub trait SPA {
    fn train_on_block<T: ?Sized>(
        &mut self,
        input: &T,
        config: &mut SPAConfig,
        train_state: &mut SPAState,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sym in input.iter() {
            loss += self.train_on_symbol(sym, config, train_state)?
        }
        Ok(loss)
    }

    fn train_on_symbol(
        &mut self,
        sym: u32,
        config: &mut SPAConfig,
        train_state: &mut SPAState,
    ) -> Result<f64>;

    fn spa_for_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn spa(
        &self,
        config: &mut SPAConfig,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Array1<f64>> {
        let mut spa = Array1::zeros(config.alphabet_size() as usize);
        for sym in 0..config.alphabet_size() {
            spa[sym as usize] = self.spa_for_symbol(sym, config, state, context_syms)?;
        }
        Ok(spa)
    }

    fn test_on_block<T: ?Sized>(
        &self,
        input: &T,
        config: &mut SPAConfig,
        inference_state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
    ) -> Result<InferenceOutput>
    where
        T: Sequence,
    {
        let mut loss: f64 = 0.;
        let mut ppl: f64 = 0.;
        let mut losses = Vec::new();
        let mut dists = Vec::new();

        let mut syms = if let Some(syms) = context_syms {
            syms.to_vec()
        } else {
            Vec::new()
        };

        syms.reserve(input.len() as usize);
        for sym in input.iter() {
            let inf_out =
                self.test_on_symbol(sym, config, inference_state, inf_out_options, Some(&syms))?;
            loss += inf_out.avg_log_loss;
            ppl += inf_out.avg_perplexity;
            losses.extend(inf_out.log_losses);
            dists.extend(inf_out.prob_dists);

            syms.push(sym);
        }
        Ok(InferenceOutput::new(
            loss / input.len() as f64,
            ppl / input.len() as f64,
            losses,
            dists,
        ))
    }

    fn test_on_symbol(
        &self,
        sym: u32,
        config: &mut SPAConfig,
        inference_state: &mut SPAState,
        inf_out_options: InfOutOptions,
        context_syms: Option<&[u32]>,
    ) -> Result<InferenceOutput>;

    fn new(config: &SPAConfig) -> Result<Self>
    where
        Self: Sized;

    fn num_symbols_seen(&self) -> u64;
}

#[derive(Debug)]
pub struct InferenceOutput {
    pub avg_log_loss: f64,
    pub avg_perplexity: f64,
    pub log_losses: Vec<f64>,
    pub prob_dists: Vec<Vec<f64>>,
}

impl InferenceOutput {
    pub fn new(
        avg_log_loss: f64,
        avg_perplexity: f64,
        log_losses: Vec<f64>,
        prob_dists: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            avg_log_loss,
            avg_perplexity,
            log_losses,
            prob_dists,
        }
    }

    pub fn into_tuple(self) -> (f64, f64, Vec<f64>, Vec<Vec<f64>>) {
        (
            self.avg_log_loss,
            self.avg_perplexity,
            self.log_losses,
            self.prob_dists,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InfOutOptions {
    Basic,
    WithLogLosses,
    Full,
}

impl InfOutOptions {
    pub fn output_losses(&self) -> bool {
        match self {
            InfOutOptions::Basic => false,
            _ => true,
        }
    }

    pub fn output_probs(&self) -> bool {
        match self {
            InfOutOptions::Full => true,
            _ => false,
        }
    }

    pub fn from_bools(output_losses: bool, output_probs: bool) -> Self {
        if output_probs {
            InfOutOptions::Full
        } else if output_losses {
            InfOutOptions::WithLogLosses
        } else {
            InfOutOptions::Basic
        }
    }
}
