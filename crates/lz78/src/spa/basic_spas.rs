use std::collections::HashMap;

use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};

use crate::{
    spa::generation::{GenerationParams, GenerationSPA},
    storage::ToFromBytes,
};

use super::{generation::gen_symbol_from_spa, SPAParams, SPA};

pub struct DirichletSPA {
    counts: HashMap<u32, u64>,
    n: u64,
}

impl SPA for DirichletSPA {
    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        let loss = -self.spa_for_symbol(input, params)?.log2();
        self.counts
            .insert(input, self.counts.get(&input).unwrap_or(&0) + 1);
        self.n += 1;
        Ok(loss)
    }

    fn spa_for_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        if let SPAParams::Dirichlet(params) = params {
            let sym_count = *self.counts.get(&sym).unwrap_or(&0) as f64;
            Ok((sym_count + params.gamma)
                / (self.n as f64 + params.gamma * params.alphabet_size as f64))
        } else {
            bail!("Wrong SPA parameters passed in for Dirichlet SPA");
        }
    }

    fn test_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64> {
        Ok(-self.spa_for_symbol(input, params)?.log2())
    }

    fn new(_params: &SPAParams) -> Result<Self> {
        Ok(Self {
            counts: HashMap::new(),
            n: 0,
        })
    }

    /// There is no state to reset
    fn reset_state(&mut self) {}

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
    fn cleanup_post_generation(&mut self) {}

    fn input_seed_data_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64> {
        self.test_on_symbol(sym, params)
    }

    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, f64)> {
        gen_symbol_from_spa(rng_sample, gen_params, &self.spa(params)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::U8Sequence;

    use super::*;

    #[test]
    fn test_dirichlet_to_from_bytes() {
        let params = SPAParams::new_dirichlet(2, 0.2);
        let mut spa = DirichletSPA::new(&params).expect("failed to make DirichletSPA");
        spa.train_on_block(
            &U8Sequence::from_data(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 2, 2, 2, 1], 3)
                .unwrap(),
            &params,
        )
        .expect("train dirichlet spa failed");

        let bytes = spa.to_bytes().expect("to bytes failed");
        let mut bytes: Bytes = bytes.into();
        let new_spa = DirichletSPA::from_bytes(&mut bytes).expect("from bytes failed");
        assert_eq!(spa.counts, new_spa.counts);
    }
}
