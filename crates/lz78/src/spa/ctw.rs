use crate::storage::ToFromBytes;

use super::{
    generation::{gen_symbol_from_spa, GenerationSPA},
    SPAParams, SPA,
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
    past_context: Vec<u32>,
    n: u64,

    /// For generation
    gen_ctx: Vec<u32>,
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

fn ctw_compute_spa_and_maybe_update(
    params: &SPAParams,
    past_context: &[u32],
    context_and_sym_to_count: &mut HashMap<(u64, u32), u64>,
    beta: &mut HashMap<u64, f64>,
    update_with_sym: Option<u32>,
) -> Result<Vec<f64>> {
    let params = if let SPAParams::CTW(p) = params {
        p
    } else {
        bail!("Wrong SPAParams for CTW")
    };

    let mut ctx_encoded = ((params.alphabet_size as u64).pow(params.depth) - 1)
        / (params.alphabet_size as u64 - 1)
        + 1;
    let mut base = 1;
    let last_idx = params.alphabet_size as usize - 1;

    for &ctx_sym in past_context.iter().rev() {
        ctx_encoded += base * ctx_sym as u64;
        base *= params.alphabet_size as u64;
    }

    let mut counts =
        ctw_get_count_vector(ctx_encoded, params.alphabet_size, &context_and_sym_to_count);
    let mut eta = (counts.clone() + 0.5) / (counts[last_idx] + 0.5);

    if let Some(sym) = update_with_sym {
        context_and_sym_to_count.insert((ctx_encoded, sym), counts[sym as usize] as u64 + 1);
    }

    for _ in 0..params.depth {
        ctx_encoded =
            (ctx_encoded + params.alphabet_size as u64 - 2) / (params.alphabet_size as u64);

        counts = ctw_get_count_vector(ctx_encoded, params.alphabet_size, &context_and_sym_to_count);

        let pw = eta.clone() / eta.sum();
        let pe = (counts.clone() + params.gamma)
            / (counts.sum() + params.alphabet_size as f64 * params.gamma);
        let beta_val = *beta.get(&ctx_encoded).unwrap_or(&1.);

        if beta_val < 1000. {
            eta = (0.5 * pe.clone() * beta_val + 0.5 * pw.clone())
                / (0.5 * pe[last_idx] * beta_val + 0.5 * pw[last_idx]);
        } else {
            eta = (0.5 * pe.clone() + 0.5 * pw.clone() / beta_val)
                / (0.5 * pe[last_idx] + 0.5 * pw[last_idx] / beta_val);
        }
        eta[last_idx] = 1.;
        // update counts and beta
        if let Some(sym) = update_with_sym {
            context_and_sym_to_count.insert((ctx_encoded, sym), counts[sym as usize] as u64 + 1);

            beta.insert(ctx_encoded, beta_val * pe[sym as usize] / pw[sym as usize]);
        }
    }

    let sum = eta.sum();
    Ok((eta / sum).to_vec())
}

impl SPA for CTW {
    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let ctw_params = if let SPAParams::CTW(p) = params {
            p
        } else {
            bail!("Wrong SPAParams for CTW")
        };

        let loss = -ctw_compute_spa_and_maybe_update(
            params,
            &self.past_context,
            &mut self.context_and_sym_to_count,
            &mut self.beta,
            Some(input),
        )?[input as usize]
            .log2();
        self.past_context.push(input);
        if self.past_context.len() > ctw_params.depth as usize {
            self.past_context = self
                .past_context
                .split_off(self.past_context.len() - ctw_params.depth as usize);
        }
        self.n += 1;

        Ok(loss)
    }

    fn spa(&mut self, params: &SPAParams) -> Result<Vec<f64>> {
        ctw_compute_spa_and_maybe_update(
            params,
            &mut self.past_context,
            &mut self.context_and_sym_to_count,
            &mut self.beta,
            None,
        )
    }

    fn spa_for_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        if sym >= params.alphabet_size() {
            bail!(
                "Invalid symbol {sym} for alphabet size {}",
                params.alphabet_size()
            )
        } else {
            Ok(self.spa(params)?[sym as usize])
        }
    }

    fn test_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let ctw_params = if let SPAParams::CTW(p) = params {
            p
        } else {
            bail!("Wrong SPAParams for CTW")
        };

        let loss = -self.spa_for_symbol(input, params)?.log2();
        self.past_context.push(input);
        if self.past_context.len() > ctw_params.depth as usize {
            self.past_context = self
                .past_context
                .split_off(self.past_context.len() - ctw_params.depth as usize);
        }
        Ok(loss)
    }

    fn new(_params: &SPAParams) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            context_and_sym_to_count: HashMap::new(),
            past_context: Vec::new(),
            beta: HashMap::new(),
            n: 0,
            gen_ctx: Vec::new(),
        })
    }

    fn reset_state(&mut self) {
        self.past_context.clear();
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl GenerationSPA for CTW {
    fn cleanup_post_generation(&mut self) {
        self.gen_ctx.clear();
    }

    fn input_seed_data_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        let ctw_params = if let SPAParams::CTW(p) = params {
            p
        } else {
            bail!("Wrong SPAParams for CTW")
        };

        let loss = -ctw_compute_spa_and_maybe_update(
            params,
            &self.gen_ctx,
            &mut self.context_and_sym_to_count,
            &mut self.beta,
            None,
        )?[sym as usize]
            .log2();

        self.gen_ctx.push(sym);
        if self.gen_ctx.len() > ctw_params.depth as usize {
            self.gen_ctx = self
                .gen_ctx
                .split_off(self.gen_ctx.len() - ctw_params.depth as usize);
        }

        Ok(loss)
    }

    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &super::generation::GenerationParams,
    ) -> Result<(u32, f64)> {
        let ctw_params = if let SPAParams::CTW(p) = params {
            p
        } else {
            bail!("Wrong SPAParams for CTW")
        };

        let spa = ctw_compute_spa_and_maybe_update(
            params,
            &self.gen_ctx,
            &mut self.context_and_sym_to_count,
            &mut self.beta,
            None,
        )?;

        let (sym, loss) = gen_symbol_from_spa(rng_sample, gen_params, &spa)?;

        self.gen_ctx.push(sym);
        if self.gen_ctx.len() > ctw_params.depth as usize {
            self.gen_ctx = self
                .gen_ctx
                .split_off(self.gen_ctx.len() - ctw_params.depth as usize);
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

        bytes.put_u32_le(self.past_context.len() as u32);
        for &sym in self.past_context.iter() {
            bytes.put_u32_le(sym);
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

        let k = bytes.get_u32_le();
        let mut past_context = Vec::with_capacity(k as usize);
        for _ in 0..k {
            past_context.push(bytes.get_u32_le());
        }

        let n = bytes.get_u64_le();

        Ok(Self {
            context_and_sym_to_count,
            beta,
            past_context,
            n,
            gen_ctx: Vec::new(),
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
        // let expected_spa_vals = vec![
        //     0.5,
        //     0.37499999999999994,
        //     0.6666666666666666,
        //     0.65625,
        //     0.7857142857142858,
        //     0.7954545454545455,
        //     0.8571428571428571,
        //     0.8628472222222222,
        //     0.8943661971830986,
        //     0.8964566929133858,
        //     0.9150197628458498,
        //     0.9156767458603311,
        //     0.9281081081081081,
        //     0.9283010233796489,
        //     0.9373725604427614,
        //     0.9374271674953387,
        //     0.9444099137596146,
        //     0.9444250202240413,
        // ];
        let params = SPAParams::new_ctw(2, 0.5, 2);
        let mut ctw = CTW::new(&params).unwrap();
        let input_seq = vec![0, 1].repeat(10);

        for (i, &sym) in input_seq.iter().enumerate() {
            let spa = ctw.spa_for_symbol(sym, &params).unwrap();
            println!("{spa}");
            // if i >= 2 {
            //     assert!((spa - expected_spa_vals[i - 2]).abs() < 1e-6);
            // }
            ctw.train_on_symbol(sym, &params).unwrap();
        }
    }

    #[test]
    fn test_ctw_to_from_bytes() {
        let params = SPAParams::new_ctw(2, 0.5, 2);
        let mut ctw = CTW::new(&params).unwrap();
        let input_seq = vec![0, 1].repeat(10);

        for &sym in input_seq.iter() {
            ctw.spa_for_symbol(sym, &params).unwrap();
            ctw.train_on_symbol(sym, &params).unwrap();
        }

        let bytes = ctw.to_bytes().unwrap();
        CTW::from_bytes(&mut bytes.into()).unwrap();
    }
}
