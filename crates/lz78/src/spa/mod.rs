use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use states::SPAState;
use util::LbAndTemp;

use crate::{sequence::Sequence, storage::ToFromBytes};

pub mod basic_spas;
pub mod causally_processed;
pub mod ctw;
pub mod generation;
pub mod lz_transform;
pub mod states;
pub mod util;

pub trait SPA {
    fn train_on_block<T: ?Sized>(
        &mut self,
        input: &T,
        params: &mut SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss = 0.0;
        for sym in input.iter() {
            loss += self.train_on_symbol(sym, params, train_state)?
        }
        Ok(loss)
    }

    fn train_on_symbol(
        &mut self,
        input: u32,
        params: &mut SPAParams,
        train_state: &mut SPAState,
    ) -> Result<f64>;

    fn spa_for_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn spa(
        &self,
        params: &mut SPAParams,
        state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<Vec<f64>> {
        let mut spa = Vec::with_capacity(params.alphabet_size() as usize);
        for sym in 0..params.alphabet_size() {
            spa.push(self.spa_for_symbol(sym, params, state, context_syms)?);
        }
        Ok(spa)
    }

    fn test_on_block<T: ?Sized>(
        &self,
        input: &T,
        params: &mut SPAParams,
        inference_state: &mut SPAState,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut loss: f64 = 0.;
        let mut syms = Vec::with_capacity(input.len() as usize);
        for sym in input.iter() {
            loss += self.test_on_symbol(sym, params, inference_state, Some(&syms))?;
            syms.push(sym);
        }
        Ok(loss)
    }

    fn test_on_symbol(
        &self,
        input: u32,
        params: &mut SPAParams,
        inference_state: &mut SPAState,
        context_syms: Option<&[u32]>,
    ) -> Result<f64>;

    fn new(params: &SPAParams) -> Result<Self>
    where
        Self: Sized;

    fn num_symbols_seen(&self) -> u64;
}

#[derive(Debug, Clone, Copy)]
pub struct DirichletSPAParams {
    alphabet_size: u32,
    pub gamma: f64,
    pub lb_and_temp: LbAndTemp,
}

#[derive(Debug, Clone, Copy)]
pub enum AdaptiveGamma {
    Inverse,
    Count,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ensemble {
    Average(u32),
    Entropy(u32),
    Depth(u32),
    None,
}

impl Ensemble {
    pub fn get_num_states(&self) -> usize {
        match &self {
            Ensemble::Average(n) => *n as usize,
            Ensemble::Entropy(n) => *n as usize,
            Ensemble::Depth(n) => *n as usize,
            Ensemble::None => 1,
        }
    }

    pub fn set_num_states(&mut self, new_n: u32) -> Result<()> {
        match self {
            Ensemble::Average(n) => *n = new_n,
            Ensemble::Entropy(n) => *n = new_n,
            Ensemble::Depth(n) => *n = new_n,
            Ensemble::None => {
                if new_n != 1 {
                    bail!("Tried to increase the number of ensemble states, but \"ensemble_type\" is \"NONE\". Set \"ensemble_type\" first.")
                }
            }
        }

        Ok(())
    }
}

impl ToFromBytes for Ensemble {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match &self {
            Ensemble::Average(n) => {
                bytes.put_u8(0);
                bytes.put_u32_le(*n);
            }
            Ensemble::Entropy(n) => {
                bytes.put_u8(1);
                bytes.put_u32_le(*n);
            }
            Ensemble::Depth(n) => {
                bytes.put_u8(2);
                bytes.put_u32_le(*n);
            }
            Ensemble::None => {
                bytes.put_u8(3);
            }
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match bytes.get_u8() {
            0 => Self::Average(bytes.get_u32_le()),
            1 => Self::Entropy(bytes.get_u32_le()),
            2 => Self::Depth(bytes.get_u32_le()),
            3 => Self::None,
            _ => bail!("uexpected ensemble type"),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackshiftParsing {
    Enabled {
        desired_context_length: u64,
        min_spa_training_points: u64,
    },
    Disabled,
}

impl BackshiftParsing {
    /// Returns a tuple of (desired_ctx_len, min_spa_training_pts), or zeros
    /// if backshift parsing is disabled
    pub fn get_params(&self) -> (u64, u64) {
        match self {
            BackshiftParsing::Enabled {
                desired_context_length,
                min_spa_training_points,
            } => (*desired_context_length, *min_spa_training_points),
            BackshiftParsing::Disabled => (0, 0),
        }
    }
}

impl ToFromBytes for BackshiftParsing {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            BackshiftParsing::Enabled {
                desired_context_length,
                min_spa_training_points,
            } => {
                bytes.put_u8(0);
                bytes.put_u64_le(*desired_context_length);
                bytes.put_u64_le(*min_spa_training_points);
            }
            BackshiftParsing::Disabled => bytes.put_u8(1),
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(match bytes.get_u8() {
            0 => {
                let desired_context_length = bytes.get_u64_le();
                let min_spa_training_points = bytes.get_u64_le();
                Self::Enabled {
                    desired_context_length,
                    min_spa_training_points,
                }
            }
            1 => Self::Disabled,
            _ => bail!("unexpected backshift parsing type"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LZ78SPAParams {
    alphabet_size: u32,
    pub inner_params: Box<SPAParams>,
    pub default_gamma: f64,
    pub adaptive_gamma: AdaptiveGamma,
    pub ensemble: Ensemble,
    pub backshift_parsing: BackshiftParsing,
    pub debug: bool,
}

impl LZ78SPAParams {
    pub fn new_dirichlet(
        alphabet_size: u32,
        gamma: f64,
        lb_and_temp: LbAndTemp,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        Self {
            alphabet_size,
            inner_params: Box::new(SPAParams::new_dirichlet(alphabet_size, gamma, lb_and_temp)),
            debug,
            default_gamma: gamma,
            adaptive_gamma,
            ensemble,
            backshift_parsing,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteThetaParams {
    pub theta_pmf: Vec<f64>,
    pub theta_values: Vec<f64>,
    alphabet_size: u32,
}

#[derive(Debug, Clone)]
pub struct CTWParams {
    alphabet_size: u32,
    pub gamma: f64,
    depth: u32,
}

#[derive(Debug, Clone)]
pub enum SPAParams {
    Dirichlet(DirichletSPAParams),
    LZ78(LZ78SPAParams),
    DiscreteTheta(DiscreteThetaParams),
    CTW(CTWParams),
}

impl SPAParams {
    pub fn get_new_state(&self) -> SPAState {
        SPAState::get_new_state(self)
    }

    pub fn new_dirichlet(alphabet_size: u32, gamma: f64, lb_and_temp: LbAndTemp) -> Self {
        Self::Dirichlet(DirichletSPAParams {
            alphabet_size,
            gamma,
            lb_and_temp,
        })
    }

    pub fn new_lz78(
        inner_spa_params: SPAParams,
        default_gamma: Option<f64>,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size: inner_spa_params.alphabet_size(),
            inner_params: Box::new(inner_spa_params),
            debug,
            default_gamma: default_gamma.unwrap_or(0.5),
            adaptive_gamma,
            ensemble,
            backshift_parsing,
        })
    }

    pub fn new_lz78_simple(inner_spa_params: SPAParams, debug: bool) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size: inner_spa_params.alphabet_size(),
            inner_params: Box::new(inner_spa_params),
            debug,
            default_gamma: 0.5,
            adaptive_gamma: AdaptiveGamma::None,
            ensemble: Ensemble::None,
            backshift_parsing: BackshiftParsing::Disabled,
        })
    }

    pub fn new_lz78_dirichlet(
        alphabet_size: u32,
        gamma: f64,
        lb_and_temp: LbAndTemp,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Box::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma,
                lb_and_temp,
            })),
            debug,
            adaptive_gamma,
            default_gamma: gamma,
            ensemble,
            backshift_parsing,
        })
    }

    pub fn default_lz78_dirichlet(alphabet_size: u32) -> Self {
        Self::LZ78(LZ78SPAParams {
            alphabet_size,
            inner_params: Box::new(Self::Dirichlet(DirichletSPAParams {
                alphabet_size,
                gamma: 0.5,
                lb_and_temp: LbAndTemp::Skip,
            })),
            debug: false,
            default_gamma: 0.5,
            adaptive_gamma: AdaptiveGamma::None,
            ensemble: Ensemble::None,
            backshift_parsing: BackshiftParsing::Disabled,
        })
    }

    pub fn new_discrete(theta_pmf: Vec<f64>, theta_values: Vec<f64>) -> Self {
        Self::DiscreteTheta(DiscreteThetaParams {
            theta_pmf,
            theta_values,
            alphabet_size: 2,
        })
    }

    pub fn new_ctw(alphabet_size: u32, gamma: f64, depth: u32) -> Self {
        Self::CTW(CTWParams {
            alphabet_size,
            gamma,
            depth,
        })
    }

    pub fn new_lz78_ctw(
        alphabet_size: u32,
        gamma: f64,
        depth: u32,
        adaptive_gamma: AdaptiveGamma,
        ensemble: Ensemble,
        backshift_parsing: BackshiftParsing,
        debug: bool,
    ) -> Self {
        let inner_params = Self::new_ctw(alphabet_size, gamma, depth);
        Self::new_lz78(
            inner_params,
            Some(gamma),
            adaptive_gamma,
            ensemble,
            backshift_parsing,
            debug,
        )
    }

    pub fn alphabet_size(&self) -> u32 {
        match self {
            SPAParams::Dirichlet(params) => params.alphabet_size,
            SPAParams::LZ78(params) => params.alphabet_size,
            SPAParams::DiscreteTheta(params) => params.alphabet_size,
            SPAParams::CTW(params) => params.alphabet_size,
        }
    }

    pub fn try_get_dirichlet(&self) -> Result<&DirichletSPAParams> {
        match &self {
            SPAParams::Dirichlet(params) => Ok(params),
            _ => bail!("Invalid SPA parameters for Dirichlet SPA"),
        }
    }

    pub fn try_get_lz78_mut(&mut self) -> Result<&mut LZ78SPAParams> {
        match self {
            SPAParams::LZ78(params) => Ok(params),
            _ => bail!("Invalid SPA parameters for LZ78 SPA"),
        }
    }

    pub fn try_get_lz78(&self) -> Result<&LZ78SPAParams> {
        match &self {
            SPAParams::LZ78(params) => Ok(params),
            _ => bail!("Invalid SPA parameters for LZ78 SPA"),
        }
    }

    pub fn try_get_ctw(&self) -> Result<&CTWParams> {
        match &self {
            SPAParams::CTW(params) => Ok(params),
            _ => bail!("Invalid SPA parameters for CTW SPA"),
        }
    }

    pub fn maybe_set_gamma(&mut self, gamma: f64) {
        match self {
            SPAParams::Dirichlet(params) => params.gamma = gamma,
            SPAParams::LZ78(params) => params.inner_params.maybe_set_gamma(gamma),
            SPAParams::DiscreteTheta(_) => {}
            SPAParams::CTW(params) => params.gamma = gamma,
        }
    }
}

impl ToFromBytes for SPAParams {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SPAParams::Dirichlet(params) => {
                bytes.put_u8(0);
                bytes.put_u32_le(params.alphabet_size);
                bytes.put_f64_le(params.gamma);
                bytes.extend(params.lb_and_temp.to_bytes()?);
            }
            SPAParams::LZ78(params) => {
                bytes.put_u8(1);
                bytes.put_u32_le(params.alphabet_size);
                bytes.put_u8(params.debug as u8);
                bytes.put_f64_le(params.default_gamma);
                bytes.put_u8(match params.adaptive_gamma {
                    AdaptiveGamma::Inverse => 0,
                    AdaptiveGamma::Count => 1,
                    AdaptiveGamma::None => 2,
                });
                bytes.extend(params.ensemble.to_bytes()?);
                bytes.extend(params.backshift_parsing.to_bytes()?);
                bytes.extend(params.inner_params.to_bytes()?);
            }
            SPAParams::DiscreteTheta(params) => {
                bytes.put_u8(2);
                bytes.put_u64_le(params.theta_pmf.len() as u64);
                for (&theta, &prob) in params.theta_values.iter().zip(params.theta_pmf.iter()) {
                    bytes.put_f64_le(theta);
                    bytes.put_f64_le(prob);
                }
            }
            SPAParams::CTW(params) => {
                bytes.put_u8(3);
                bytes.put_u32_le(params.alphabet_size);
                bytes.put_f64_le(params.gamma);
                bytes.put_u32_le(params.depth);
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
                let lb_and_temp = LbAndTemp::from_bytes(bytes)?;
                Ok({
                    Self::Dirichlet(DirichletSPAParams {
                        alphabet_size,
                        gamma,
                        lb_and_temp,
                    })
                })
            }
            1 => {
                let alphabet_size = bytes.get_u32_le();
                let debug = bytes.get_u8() == 1;
                let default_gamma = bytes.get_f64_le();
                let adaptive_gamma = match bytes.get_u8() {
                    0 => AdaptiveGamma::Inverse,
                    1 => AdaptiveGamma::Count,
                    2 => AdaptiveGamma::None,
                    _ => bail!("Unexpected AdaptiveGamma type"),
                };
                let ensemble = Ensemble::from_bytes(bytes)?;
                let backshift_parsing = BackshiftParsing::from_bytes(bytes)?;
                let inner_params = Self::from_bytes(bytes)?;
                Ok(Self::LZ78(LZ78SPAParams {
                    alphabet_size,
                    inner_params: Box::new(inner_params),
                    debug,
                    default_gamma,
                    adaptive_gamma,
                    ensemble,
                    backshift_parsing,
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
            3 => {
                let alphabet_size = bytes.get_u32_le();
                let gamma = bytes.get_f64_le();
                let depth = bytes.get_u32_le();
                Ok(Self::new_ctw(alphabet_size, gamma, depth))
            }
            _ => bail!("Unexpected SPA type indicator {tpe}"),
        }
    }
}
