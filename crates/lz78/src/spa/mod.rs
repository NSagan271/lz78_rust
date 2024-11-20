use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use std::sync::Arc;

use crate::{sequence::Sequence, storage::ToFromBytes};

pub mod basic_spas;
pub mod causally_processed;
pub mod generation;
pub mod lz_transform;

pub trait SPA {
    fn train_on_block<T: ?Sized>(&mut self, input: &T, params: &SPAParams) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sym in input.iter() {
            loss += self.train_on_symbol(sym, params)?
        }
        Ok(loss)
    }

    fn train_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64>;

    fn spa_for_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64>;

    fn spa(&mut self, params: &SPAParams) -> Result<Vec<f64>> {
        let mut spa = Vec::with_capacity(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa.push(self.spa_for_symbol(sym, params)?);
        }
        Ok(spa)
    }

    fn test_on_block<T: ?Sized>(&mut self, input: &T, params: &SPAParams) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss: f64 = 0.;
        for sym in input.iter() {
            loss += self.test_on_symbol(sym, params)?;
        }
        Ok(loss)
    }

    fn test_on_symbol(&mut self, input: u32, params: &SPAParams) -> Result<f64>;

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized;

    fn reset_state(&mut self);

    fn num_symbols_seen(&self) -> u64;
}

#[derive(Debug, Clone, Copy)]
pub struct DirichletSPAParams {
    alphabet_size: u32,
    gamma: f64,
}

#[derive(Debug, Clone)]
pub struct LZ78SPAParams {
    alphabet_size: u32,
    pub inner_params: Arc<SPAParams>,
    pub debug: bool,
}

#[derive(Debug, Clone)]
pub struct DiscreteThetaParams {
    pub theta_pmf: Vec<f64>,
    pub theta_values: Vec<f64>,
    alphabet_size: u32,
}

#[derive(Debug, Clone)]
pub enum SPAParams {
    Dirichlet(DirichletSPAParams),
    LZ78(LZ78SPAParams),
    DiscreteTheta(DiscreteThetaParams),
}

impl SPAParams {
    pub fn new_dirichlet(alphabet_size: u32, gamma: f64) -> Self {
        Self::Dirichlet(DirichletSPAParams {
            alphabet_size,
            gamma,
        })
    }

    pub fn new_lz78(inner_spa_params: SPAParams, debug: bool) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size: inner_spa_params.alphabet_size(),
            inner_params: Arc::new(inner_spa_params),
            debug,
        })
    }

    pub fn new_lz78_dirichlet(alphabet_size: u32, gamma: f64, debug: bool) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Arc::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma,
            })),
            debug,
        })
    }

    pub fn default_lz78_dirichlet(alphabet_size: u32) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Arc::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma: 0.5,
            })),
            debug: false,
        })
    }

    pub fn new_discrete(theta_pmf: Vec<f64>, theta_values: Vec<f64>) -> Self {
        Self::DiscreteTheta(DiscreteThetaParams {
            theta_pmf,
            theta_values,
            alphabet_size: 2,
        })
    }

    pub fn alphabet_size(&self) -> u32 {
        match self {
            SPAParams::Dirichlet(params) => params.alphabet_size,
            SPAParams::LZ78(params) => params.alphabet_size,
            SPAParams::DiscreteTheta(params) => params.alphabet_size,
        }
    }
}

impl ToFromBytes for SPAParams {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SPAParams::Dirichlet(dirichlet_spaparams) => {
                bytes.put_u8(0);
                bytes.put_u32_le(dirichlet_spaparams.alphabet_size);
                bytes.put_f64_le(dirichlet_spaparams.gamma);
            }
            SPAParams::LZ78(lz78_spaparams) => {
                bytes.put_u8(1);
                bytes.put_u32_le(lz78_spaparams.alphabet_size);
                bytes.put_u8(lz78_spaparams.debug as u8);
                bytes.extend(lz78_spaparams.inner_params.to_bytes()?);
            }
            SPAParams::DiscreteTheta(discrete_theta_params) => {
                bytes.put_u8(2);
                bytes.put_u64_le(discrete_theta_params.theta_pmf.len() as u64);
                for (&theta, &prob) in discrete_theta_params
                    .theta_values
                    .iter()
                    .zip(discrete_theta_params.theta_pmf.iter())
                {
                    bytes.put_f64_le(theta);
                    bytes.put_f64_le(prob);
                }
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let tpe = bytes.get_u8();
        match tpe {
            0 => {
                let alphabet_size = bytes.get_u32_le();
                let gamma = bytes.get_f64_le();
                Ok({
                    Self::Dirichlet(DirichletSPAParams {
                        alphabet_size,
                        gamma,
                    })
                })
            }
            1 => {
                let alphabet_size = bytes.get_u32_le();
                let debug = bytes.get_u8() == 1;
                let inner_params = Self::from_bytes(bytes)?;
                Ok(Self::LZ78(LZ78SPAParams {
                    alphabet_size,
                    inner_params: Arc::new(inner_params),
                    debug,
                }))
            }
            2 => {
                let n = bytes.get_u64_le();

                let mut theta_values: Vec<f64> = Vec::with_capacity(n as usize);
                let mut theta_pmf: Vec<f64> = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    theta_values.push(bytes.get_f64_le());
                    theta_pmf.push(bytes.get_f64_le());
                }

                Ok(Self::new_discrete(theta_pmf, theta_values))
            }
            _ => bail!("Unexpected SPA type indicator {tpe}"),
        }
    }
}
