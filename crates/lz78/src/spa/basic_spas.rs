use std::collections::HashMap;

use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};

use crate::{
    spa::generation::{GenerationParams, GenerationSPA},
    storage::ToFromBytes,
};

use super::{generation::gen_symbol_from_spa, states::SPAState, SPAParams, SPA};

pub struct DirichletSPA {
    counts: HashMap<u32, u64>,
    n: u64,
}

impl SPA for DirichletSPA {
    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &mut SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64> {
        let loss = -self
            .spa_for_symbol(input, params, train_state, None)?
            .log2();
        self.counts
            .insert(input, self.counts.get(&input).unwrap_or(&0) + 1);
        self.n += 1;
        Ok(loss)
    }

    // TODO: add lb and temperature
    fn spa_for_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        _train_state: &mut SPAState,
        _context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        let params = params.try_get_dirichlet()?;
        let sym_count = *self.counts.get(&sym).unwrap_or(&0) as f64;
        Ok((sym_count + params.gamma)
            / (self.n as f64 + params.gamma * params.alphabet_size as f64))
    }

    fn test_on_symbol(
        &self,
        input: u32,
        params: &mut SPAParams,
        inference_state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64> {
        Ok(-self
            .spa_for_symbol(input, params, inference_state, context_syms)?
            .log2())
    }

    fn new(_params: &SPAParams) -> Result<Self> {
        Ok(Self {
            counts: HashMap::new(),
            n: 0,
        })
    }

    fn num_symbols_seen(&self) -> u64 {
        self.n
    }
}

impl ToFromBytes for DirichletSPA {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.counts.len() as u32);
        for (&sym, &count) in self.counts.iter() {
            bytes.put_u32_le(sym);
            bytes.put_u64_le(count);
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let counts_len = bytes.get_u32_le();

        let mut counts: HashMap<u32, u64> = HashMap::new();
        let mut n = 0;
        for _ in 0..counts_len {
            let sym = bytes.get_u32_le();
            let count = bytes.get_u64_le();
            n += count;
            counts.insert(sym, count);
        }

        Ok(Self { counts, n })
    }
}

impl GenerationSPA for DirichletSPA {
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        gen_state: &mut SPAState,
    ) -> Result<f64> {
        self.test_on_symbol(sym, params, gen_state, None)
    }

    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &mut SPAParams,
        gen_params: &GenerationParams,
        gen_state: &mut SPAState,
        _context_syms: &[u32],
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(rng_sample, gen_params, &self.spa(params, gen_state, None)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::U8Sequence;

    use super::*;

    #[test]
    fn test_dirichlet_to_from_bytes() {
        let mut params = SPAParams::new_dirichlet(2, 0.2);
        let mut state = SPAState::None;
        let mut spa = DirichletSPA::new(&params).expect("failed to make DirichletSPA");
        spa.train_on_block(
            &U8Sequence::from_data(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 2, 2, 2, 1], 3)
                .unwrap(),
            &mut params,
            &mut state,
        )
        .expect("train dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPA::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.counts, new_spa.counts);
    }
}
