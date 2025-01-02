use crate::storage::ToFromBytes;

use super::{
    generation::{gen_symbol_from_spa, GenerationParams, GenerationSPA},
    states::SPAState,
    CTWParams, SPAParams, SPA,
};
use anyhow::{bail, Ok, Result};
use bytes::{Buf, BufMut};
use ndarray::Array1;
use std::collections::HashMap;

//   Translated from Python written by Eli Pugh and Ethan Shen
//   https://github.com/elipugh/directed_information
//
//   Translated from Matlab written by Jiantao Jiao
//   https://github.com/EEthinker/Universal_directed_information
//
//   Based off of:
//     F. Willems, Y. Shtarkov and T. Tjalkens
//     'The context-tree weighting method: basic properties'
//     https://ieeexplore.ieee.org/document/382012

pub struct CTW {
    context_and_sym_to_count: HashMap<(u64, u32), u64>,
    beta: HashMap<u64, f64>,
    n: u64,
}

fn ctw_get_count_vector(
    ctx_encoded: u64,
    alpha_size: u32,
    context_and_sym_to_count: &HashMap<(u64, u32), u64>,
) -> Array1<f64> {
    let mut arr = Array1::zeros(alpha_size as usize);
    for sym in 0..alpha_size {
        arr[sym as usize] = *context_and_sym_to_count
            .get(&(ctx_encoded, sym))
            .unwrap_or(&0) as f64;
    }

    arr
}

fn _get_initial_ctx_encoded_counts_and_eta(
    params: &CTWParams,
    past_context: &[u32],
    context_and_sym_to_count: &HashMap<(u64, u32), u64>,
) -> (u64, Array1<f64>, Array1<f64>) {
    let mut ctx_encoded = ((params.alphabet_size as u64).pow(params.depth) - 1)
        / (params.alphabet_size as u64 - 1)
        + 1;
    let mut base = 1;
    let last_idx = params.alphabet_size as usize - 1;

    for &ctx_sym in past_context.iter().rev() {
        ctx_encoded += base * ctx_sym as u64;
        base *= params.alphabet_size as u64;
    }

    let counts = ctw_get_count_vector(ctx_encoded, params.alphabet_size, &context_and_sym_to_count);
    let eta = (counts.clone() + 0.5) / (counts[last_idx] + 0.5);

    (ctx_encoded, counts, eta)
}

fn _ctw_inner_loop(
    ctx_encoded: &mut u64,
    last_idx: usize,
    params: &CTWParams,
    counts: &mut Array1<f64>,
    context_and_sym_to_count: &HashMap<(u64, u32), u64>,
    eta: &mut Array1<f64>,
    beta: &HashMap<u64, f64>,
) -> (f64, Array1<f64>, Array1<f64>) {
    *ctx_encoded = (*ctx_encoded + params.alphabet_size as u64 - 2) / (params.alphabet_size as u64);

    *counts = ctw_get_count_vector(
        *ctx_encoded,
        params.alphabet_size,
        &context_and_sym_to_count,
    );

    let pw = eta.clone() / eta.sum();
    let pe = (counts.clone() + params.gamma)
        / (counts.sum() + params.alphabet_size as f64 * params.gamma);
    let beta_val = *beta.get(&ctx_encoded).unwrap_or(&1.);

    if beta_val < 1000. {
        *eta = (0.5 * pe.clone() * beta_val + 0.5 * pw.clone())
            / (0.5 * pe[last_idx] * beta_val + 0.5 * pw[last_idx]);
    } else {
        *eta = (0.5 * pe.clone() + 0.5 * pw.clone() / beta_val)
            / (0.5 * pe[last_idx] + 0.5 * pw[last_idx] / beta_val);
    }
    eta[last_idx] = 1.;

    (beta_val, pe, pw)
}

fn ctw_compute_spa(
    params: &SPAParams,
    past_context: &[u32],
    context_and_sym_to_count: &HashMap<(u64, u32), u64>,
    beta: &HashMap<u64, f64>,
) -> Result<Vec<f64>> {
    let params = params.try_get_ctw()?;
    let last_idx = params.alphabet_size as usize - 1;
    let (mut ctx_encoded, mut counts, mut eta) =
        _get_initial_ctx_encoded_counts_and_eta(params, past_context, context_and_sym_to_count);

    for _ in 0..params.depth {
        _ctw_inner_loop(
            &mut ctx_encoded,
            last_idx,
            params,
            &mut counts,
            context_and_sym_to_count,
            &mut eta,
            beta,
        );
    }

    let sum = eta.sum();
    Ok((eta / sum).to_vec())
}

fn ctw_compute_spa_and_maybe_update(
    params: &SPAParams,
    past_context: &[u32],
    context_and_sym_to_count: &mut HashMap<(u64, u32), u64>,
    beta: &mut HashMap<u64, f64>,
    update_with_sym: Option<u32>,
) -> Result<Vec<f64>> {
    let params = params.try_get_ctw()?;
    let last_idx = params.alphabet_size as usize - 1;
    let (mut ctx_encoded, mut counts, mut eta) =
        _get_initial_ctx_encoded_counts_and_eta(params, past_context, context_and_sym_to_count);

    if let Some(sym) = update_with_sym {
        context_and_sym_to_count.insert((ctx_encoded, sym), counts[sym as usize] as u64 + 1);
    }

    for _ in 0..params.depth {
        let (beta_val, pe, pw) = _ctw_inner_loop(
            &mut ctx_encoded,
            last_idx,
            params,
            &mut counts,
            context_and_sym_to_count,
            &mut eta,
            beta,
        );
        if let Some(sym) = update_with_sym {
            context_and_sym_to_count.insert((ctx_encoded, sym), counts[sym as usize] as u64 + 1);

            beta.insert(ctx_encoded, beta_val * pe[sym as usize] / pw[sym as usize]);
        }
    }

    let sum = eta.sum();
    Ok((eta / sum).to_vec())
}

impl SPA for CTW {
    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64> {
        let ctw_params = params.try_get_ctw()?;
        let state = train_state.try_get_ctw()?;

        let loss = -ctw_compute_spa_and_maybe_update(
            params,
            &state.context,
            &mut self.context_and_sym_to_count,
            &mut self.beta,
            if state.context.len() >= ctw_params.depth as usize {
                Some(input)
            } else {
                None
            },
        )?[input as usize]
            .log2();
        state.context.push(input);
        if state.context.len() > ctw_params.depth as usize {
            state.context = state
                .context
                .split_off(state.context.len() - ctw_params.depth as usize);
        }
        self.n += 1;

        Ok(loss)
    }

    fn spa(&self, params: &SPAParams, state: &mut SPAState) -> Result<Vec<f64>> {
        let state = state.try_get_ctw()?;
        ctw_compute_spa(
            params,
            &state.context,
            &self.context_and_sym_to_count,
            &self.beta,
        )
    }

    fn spa_for_symbol(&self, sym: u32, params: &SPAParams, state: &mut SPAState) -> Result<f64> {
        if sym >= params.alphabet_size() {
            bail!(
                "Invalid symbol {sym} for alphabet size {}",
                params.alphabet_size()
            )
        } else {
            Ok(self.spa(params, state)?[sym as usize])
        }
    }

    fn test_on_symbol(
        &self,
        input: u32,
        params: &SPAParams,
        inference_state: &mut SPAState,
    ) -> Result<f64> {
        let ctw_params = params.try_get_ctw()?;
        let loss = -self.spa_for_symbol(input, params, inference_state)?.log2();

        let inference_state = inference_state.try_get_ctw()?;
        inference_state.context.push(input);
        if inference_state.context.len() > ctw_params.depth as usize {
            inference_state.context = inference_state
                .context
                .split_off(inference_state.context.len() - ctw_params.depth as usize);
        }
        Ok(loss)
    }

    fn new(_params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            context_and_sym_to_count: HashMap::new(),
            beta: HashMap::new(),
            n: 0,
        })
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl GenerationSPA for CTW {
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &SPAParams,
        gen_state: &mut SPAState,
    ) -> Result<f64> {
        let ctw_params = params.try_get_ctw()?;
        let gen_state = gen_state.try_get_ctw()?;

        let loss = -ctw_compute_spa(
            params,
            &gen_state.context,
            &self.context_and_sym_to_count,
            &self.beta,
        )?[sym as usize]
            .log2();

        gen_state.context.push(sym);
        if gen_state.context.len() > ctw_params.depth as usize {
            gen_state.context = gen_state
                .context
                .split_off(gen_state.context.len() - ctw_params.depth as usize);
        }

        Ok(loss)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &GenerationParams,
        gen_state: &mut SPAState,
    ) -> Result<(u32, f64)> {
        let ctw_params = params.try_get_ctw()?;
        let gen_state = gen_state.try_get_ctw()?;

        let spa = ctw_compute_spa(
            params,
            &gen_state.context,
            &self.context_and_sym_to_count,
            &self.beta,
        )?;

        let (sym, loss) = gen_symbol_from_spa(rng_sample, gen_params, &spa)?;

        gen_state.context.push(sym);
        if gen_state.context.len() > ctw_params.depth as usize {
            gen_state.context = gen_state
                .context
                .split_off(gen_state.context.len() - ctw_params.depth as usize);
        }

        Ok((sym, loss))
    }
}

impl ToFromBytes for CTW {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.context_and_sym_to_count.len() as u64);
        for (&(ctx_val, sym), &cnt) in self.context_and_sym_to_count.iter() {
            bytes.put_u64_le(ctx_val);
            bytes.put_u32_le(sym);
            bytes.put_u64_le(cnt);
        }

        bytes.put_u64_le(self.beta.len() as u64);
        for (&ctx_val, &beta) in self.beta.iter() {
            bytes.put_u64_le(ctx_val);
            bytes.put_f64_le(beta);
        }

        bytes.put_u64_le(self.n);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let k = bytes.get_u64_le();
        let mut context_and_sym_to_count = HashMap::with_capacity(k as usize);
        for _ in 0..k {
            let (ctx_val, sym, cnt) = (bytes.get_u64_le(), bytes.get_u32_le(), bytes.get_u64_le());
            context_and_sym_to_count.insert((ctx_val, sym), cnt);
        }

        let k = bytes.get_u64_le();
        let mut beta = HashMap::with_capacity(k as usize);
        for _ in 0..k {
            let (ctx_val, b) = (bytes.get_u64_le(), bytes.get_f64_le());
            beta.insert(ctx_val, b);
        }

        let n = bytes.get_u64_le();

        Ok(Self {
            context_and_sym_to_count,
            beta,
            n,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        spa::{SPAParams, SPA},
        storage::ToFromBytes,
    };

    use super::CTW;

    #[test]
    fn test_ctw_against_python() {
        // SPA values python implementation
        let expected_spa_vals = vec![
            0.5,
            0.37499999999999994,
            0.6666666666666666,
            0.65625,
            0.7857142857142858,
            0.7954545454545455,
            0.8571428571428571,
            0.8628472222222222,
            0.8943661971830986,
            0.8964566929133858,
            0.9150197628458498,
            0.9156767458603311,
            0.9281081081081081,
            0.9283010233796489,
            0.9373725604427614,
            0.9374271674953387,
            0.9444099137596146,
            0.9444250202240413,
        ];
        let params = SPAParams::new_ctw(2, 0.5, 2);
        let mut state = params.get_new_state(false);
        let mut ctw = CTW::new(&params).unwrap();
        let input_seq = vec![0, 1].repeat(10);

        for (i, &sym) in input_seq.iter().enumerate() {
            let spa = ctw.spa_for_symbol(sym, &params, &mut state).unwrap();
            // println!("{spa}");
            if i >= 2 {
                assert!((spa - expected_spa_vals[i - 2]).abs() < 1e-6);
            }
            ctw.train_on_symbol(sym, &params, &mut state).unwrap();
        }
    }

    #[test]
    fn test_ctw_to_from_bytes() {
        let params = SPAParams::new_ctw(2, 0.5, 2);
        let mut state = params.get_new_state(false);
        let mut ctw = CTW::new(&params).unwrap();
        let input_seq = vec![0, 1].repeat(10);

        for &sym in input_seq.iter() {
            ctw.train_on_symbol(sym, &params, &mut state).unwrap();
        }

        let bytes = ctw.to_bytes().unwrap();
        CTW::from_bytes(&mut bytes.into()).unwrap();
    }
}
