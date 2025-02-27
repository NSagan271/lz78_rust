use crate::storage::ToFromBytes;

use super::{states::SPAState, util::LbAndTemp};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub enum SPAParams {
    Dirichlet(DirichletParams),
    LZ78(LZ78Params),
    Discrete(DiscreteThetaParams),
    DiricDirichlet(DiracDirichletParams),
}

impl ToFromBytes for SPAParams {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            SPAParams::Dirichlet(params) => {
                bytes.put_u8(0);
                bytes.extend(params.to_bytes()?);
            }
            SPAParams::LZ78(params) => {
                bytes.put_u8(1);
                bytes.extend(params.to_bytes()?);
            }
            SPAParams::Discrete(params) => {
                bytes.put_u8(2);
                bytes.extend(params.to_bytes()?);
            }
            SPAParams::DiricDirichlet(params) => {
                bytes.put_u8(3);
                bytes.extend(params.to_bytes()?);
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            0 => Ok(Self::Dirichlet(DirichletParams::from_bytes(bytes)?)),
            1 => Ok(Self::LZ78(LZ78Params::from_bytes(bytes)?)),
            2 => Ok(Self::Discrete(DiscreteThetaParams::from_bytes(bytes)?)),
            3 => Ok(Self::DiricDirichlet(DiracDirichletParams::from_bytes(
                bytes,
            )?)),
            _ => bail!("Could not decode SPAParams from bytes"),
        }
    }
}

impl SPAParams {
    pub fn try_get_dirichlet_mut(&mut self) -> Result<&mut DirichletParams> {
        if let SPAParams::Dirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirichlet SPA Info.")
        }
    }
    pub fn try_get_dirichlet(&self) -> Result<&DirichletParams> {
        if let SPAParams::Dirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirichlet SPA Info.")
        }
    }

    pub fn try_get_lz78_mut(&mut self) -> Result<&mut LZ78Params> {
        if let SPAParams::LZ78(info) = self {
            Ok(info)
        } else {
            bail!("Expected LZ78 SPA Info.")
        }
    }

    pub fn try_get_lz78(&self) -> Result<&LZ78Params> {
        if let SPAParams::LZ78(info) = self {
            Ok(info)
        } else {
            bail!("Expected LZ78 SPA Info.")
        }
    }

    pub fn try_get_discrete_mut(&mut self) -> Result<&mut DiscreteThetaParams> {
        if let SPAParams::Discrete(info) = self {
            Ok(info)
        } else {
            bail!("Expected Discrete SPA Info.")
        }
    }

    pub fn try_get_discrete(&self) -> Result<&DiscreteThetaParams> {
        if let SPAParams::Discrete(info) = self {
            Ok(info)
        } else {
            bail!("Expected Discrete SPA Info.")
        }
    }

    pub fn try_get_dirac_mut(&mut self) -> Result<&mut DiracDirichletParams> {
        if let SPAParams::DiricDirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirac-Dirichlet SPA Info.")
        }
    }

    pub fn try_get_dirac(&self) -> Result<&DiracDirichletParams> {
        if let SPAParams::DiricDirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirac-Dirichlet SPA Info.")
        }
    }

    pub fn alphabet_size(&self) -> u32 {
        match self {
            SPAParams::Dirichlet(info) => info.alphabet_size,
            SPAParams::LZ78(info) => info.inner_params.alphabet_size(),
            SPAParams::Discrete(_) => 2,
            SPAParams::DiricDirichlet(_) => 2,
        }
    }

    pub fn maybe_get_gamma(&self) -> Option<f64> {
        match self {
            SPAParams::Dirichlet(info) => Some(info.gamma),
            _ => None,
        }
    }

    pub fn maybe_set_gamma(&mut self, gamma: f64) {
        match self {
            SPAParams::Dirichlet(info) => info.gamma = gamma,
            _ => {}
        }
    }

    pub fn get_new_state(&self) -> SPAState {
        SPAState::get_new_state(self)
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteThetaParams {
    pub theta_pmf: Array1<f64>,
    pub theta_values: Array1<f64>,
}

impl ToFromBytes for DiscreteThetaParams {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.theta_pmf.len() as u64);
        for (&theta, &prob) in self.theta_values.iter().zip(self.theta_pmf.iter()) {
            bytes.put_f64_le(theta);
            bytes.put_f64_le(prob);
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le();

        let mut theta_values: Vec<f64> = Vec::with_capacity(n as usize);
        let mut theta_pmf: Vec<f64> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            theta_values.push(bytes.get_f64_le());
            theta_pmf.push(bytes.get_f64_le());
        }

        Ok(Self {
            theta_pmf: Array1::from_vec(theta_pmf),
            theta_values: Array1::from_vec(theta_values),
        })
    }
}

impl DiscreteThetaParams {
    pub fn new(theta_pmf: &[f64], theta_values: &[f64]) -> Self {
        Self {
            theta_pmf: Array1::from_vec(theta_pmf.to_vec()),
            theta_values: Array1::from_vec(theta_values.to_vec()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiracDirichletParams {
    pub dirichlet_params: Box<SPAParams>,
    pub disc_params: DiscreteThetaParams,
    pub dirichlet_weight: f64,
}

impl ToFromBytes for DiracDirichletParams {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.dirichlet_params.to_bytes()?;
        bytes.extend(self.disc_params.to_bytes()?);
        bytes.put_f64_le(self.dirichlet_weight);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let dirichlet_params = Box::new(SPAParams::from_bytes(bytes)?);
        let disc_params = DiscreteThetaParams::from_bytes(bytes)?;
        let dirichlet_weight = bytes.get_f64_le();
        Ok(Self {
            dirichlet_params,
            disc_params,
            dirichlet_weight,
        })
    }
}

impl DiracDirichletParams {
    pub fn new(theta_pmf: &[f64], theta_values: &[f64], gamma: f64, dirichlet_weight: f64) -> Self {
        Self {
            dirichlet_params: Box::new(SPAParams::Dirichlet(DirichletParams {
                gamma,
                alphabet_size: 2,
                lb_and_temp: LbAndTemp::Skip,
            })),
            disc_params: DiscreteThetaParams::new(theta_pmf, theta_values),
            dirichlet_weight,
        }
    }
    pub fn new_enum(
        theta_pmf: &[f64],
        theta_values: &[f64],
        gamma: f64,
        dirichlet_weight: f64,
    ) -> SPAParams {
        SPAParams::DiricDirichlet(Self::new(theta_pmf, theta_values, gamma, dirichlet_weight))
    }
}

#[derive(Debug, Clone)]
pub struct DirichletParams {
    pub gamma: f64,
    pub alphabet_size: u32,
    pub lb_and_temp: LbAndTemp,
}

impl ToFromBytes for DirichletParams {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_f64_le(self.gamma);
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.lb_and_temp.to_bytes()?);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let gamma = bytes.get_f64_le();
        let alphabet_size = bytes.get_u32_le();
        let lb_and_temp = LbAndTemp::from_bytes(bytes)?;
        Ok(Self {
            gamma,
            alphabet_size,
            lb_and_temp,
        })
    }
}

pub struct DirichletParamsBuilder {
    gamma: f64,
    alphabet_size: u32,
    lb_and_temp: LbAndTemp,
}

impl DirichletParamsBuilder {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            alphabet_size,
            gamma: 0.5,
            lb_and_temp: LbAndTemp::Skip,
        }
    }

    pub fn gamma(&mut self, gamma: f64) -> &mut Self {
        self.gamma = gamma;
        self
    }

    pub fn lb_and_temp(&mut self, lb: f64, temp: f64, lb_first: bool) -> &mut Self {
        if lb_first {
            self.lb_and_temp = LbAndTemp::LbFirst { lb, temp };
        } else {
            self.lb_and_temp = LbAndTemp::TempFirst { lb, temp }
        }

        self
    }

    pub fn build(&self) -> DirichletParams {
        DirichletParams {
            alphabet_size: self.alphabet_size,
            gamma: self.gamma,
            lb_and_temp: self.lb_and_temp,
        }
    }

    pub fn build_enum(&self) -> SPAParams {
        SPAParams::Dirichlet(self.build())
    }
}

#[derive(Debug, Clone)]
pub struct LZ78Params {
    pub inner_params: Box<SPAParams>,
    pub adaptive_gamma: AdaptiveGamma,
    pub ensemble: Ensemble,
    pub par_ensemble: bool,
    pub backshift_parsing: BackshiftParsing,
    pub debug: bool,
}

impl ToFromBytes for LZ78Params {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.inner_params.to_bytes()?;
        bytes.extend(self.adaptive_gamma.to_bytes()?);
        bytes.extend(self.ensemble.to_bytes()?);
        bytes.put_u8(self.par_ensemble as u8);
        bytes.extend(self.backshift_parsing.to_bytes()?);
        bytes.put_u8(self.debug as u8);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let inner_params = Box::new(SPAParams::from_bytes(bytes)?);
        let adaptive_gamma = AdaptiveGamma::from_bytes(bytes)?;
        let ensemble = Ensemble::from_bytes(bytes)?;
        let par_ensemble = bytes.get_u8() > 0;
        let backshift_parsing = BackshiftParsing::from_bytes(bytes)?;
        let debug = bytes.get_u8() > 0;
        Ok(Self {
            inner_params,
            adaptive_gamma,
            ensemble,
            par_ensemble,
            backshift_parsing,
            debug,
        })
    }
}

pub struct LZ78ParamsBuilder {
    inner_params: Box<SPAParams>,
    adaptive_gamma: AdaptiveGamma,
    ensemble: Ensemble,
    par_ensemble: bool,
    backshift_parsing: BackshiftParsing,
    debug: bool,
}

impl LZ78ParamsBuilder {
    pub fn new(inner_info: SPAParams) -> Self {
        Self {
            inner_params: Box::new(inner_info),
            adaptive_gamma: AdaptiveGamma::None,
            ensemble: Ensemble::None,
            par_ensemble: false,
            backshift_parsing: BackshiftParsing::Disabled,
            debug: false,
        }
    }

    pub fn adaptive_gamma(mut self, adaptive_gamma: AdaptiveGamma) -> Self {
        self.adaptive_gamma = adaptive_gamma;
        self
    }

    pub fn ensemble(mut self, ensemble: Ensemble, parallel: bool) -> Self {
        self.ensemble = ensemble;
        self.par_ensemble = parallel;
        self
    }

    pub fn backshift(mut self, desired_context_length: u64, min_spa_training_points: u64) -> Self {
        self.backshift_parsing = BackshiftParsing::Enabled {
            desired_context_length,
            min_spa_training_points,
        };
        self
    }

    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn build(self) -> LZ78Params {
        LZ78Params {
            inner_params: self.inner_params,
            adaptive_gamma: self.adaptive_gamma,
            ensemble: self.ensemble,
            par_ensemble: self.par_ensemble,
            backshift_parsing: self.backshift_parsing,
            debug: self.debug,
        }
    }

    pub fn build_enum(self) -> SPAParams {
        SPAParams::LZ78(self.build())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AdaptiveGamma {
    Inverse,
    Count,
    None,
}

impl ToFromBytes for AdaptiveGamma {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        match self {
            AdaptiveGamma::Inverse => Ok(vec![0]),
            AdaptiveGamma::Count => Ok(vec![1]),
            AdaptiveGamma::None => Ok(vec![2]),
        }
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            0 => Ok(Self::Inverse),
            1 => Ok(Self::Count),
            2 => Ok(Self::None),
            _ => bail!("Failed to decode AdaptiveGamma from bytes"),
        }
    }
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
