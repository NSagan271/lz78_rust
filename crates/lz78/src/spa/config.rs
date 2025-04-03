use crate::storage::ToFromBytes;

use super::{states::SPAState, util::LbAndTemp};
use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub enum SPAConfig {
    Dirichlet(DirichletConfig),
    LZ78(LZ78Config),
    Discrete(DiscreteThetaConfig),
    DiricDirichlet(DiracDirichletConfig),
}

impl ToFromBytes for SPAConfig {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            SPAConfig::Dirichlet(config) => {
                bytes.put_u8(0);
                bytes.extend(config.to_bytes()?);
            }
            SPAConfig::LZ78(config) => {
                bytes.put_u8(1);
                bytes.extend(config.to_bytes()?);
            }
            SPAConfig::Discrete(config) => {
                bytes.put_u8(2);
                bytes.extend(config.to_bytes()?);
            }
            SPAConfig::DiricDirichlet(config) => {
                bytes.put_u8(3);
                bytes.extend(config.to_bytes()?);
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        match bytes.get_u8() {
            0 => Ok(Self::Dirichlet(DirichletConfig::from_bytes(bytes)?)),
            1 => Ok(Self::LZ78(LZ78Config::from_bytes(bytes)?)),
            2 => Ok(Self::Discrete(DiscreteThetaConfig::from_bytes(bytes)?)),
            3 => Ok(Self::DiricDirichlet(DiracDirichletConfig::from_bytes(
                bytes,
            )?)),
            _ => bail!("Could not decode SPAConfig from bytes"),
        }
    }
}

impl SPAConfig {
    pub fn try_get_dirichlet_mut(&mut self) -> Result<&mut DirichletConfig> {
        if let SPAConfig::Dirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirichlet SPA Info.")
        }
    }
    pub fn try_get_dirichlet(&self) -> Result<&DirichletConfig> {
        if let SPAConfig::Dirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirichlet SPA Info.")
        }
    }

    pub fn try_get_lz78_mut(&mut self) -> Result<&mut LZ78Config> {
        if let SPAConfig::LZ78(info) = self {
            Ok(info)
        } else {
            bail!("Expected LZ78 SPA Info.")
        }
    }

    pub fn try_get_lz78(&self) -> Result<&LZ78Config> {
        if let SPAConfig::LZ78(info) = self {
            Ok(info)
        } else {
            bail!("Expected LZ78 SPA Info.")
        }
    }

    pub fn try_get_discrete_mut(&mut self) -> Result<&mut DiscreteThetaConfig> {
        if let SPAConfig::Discrete(info) = self {
            Ok(info)
        } else {
            bail!("Expected Discrete SPA Info.")
        }
    }

    pub fn try_get_discrete(&self) -> Result<&DiscreteThetaConfig> {
        if let SPAConfig::Discrete(info) = self {
            Ok(info)
        } else {
            bail!("Expected Discrete SPA Info.")
        }
    }

    pub fn try_get_dirac_mut(&mut self) -> Result<&mut DiracDirichletConfig> {
        if let SPAConfig::DiricDirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirac-Dirichlet SPA Info.")
        }
    }

    pub fn try_get_dirac(&self) -> Result<&DiracDirichletConfig> {
        if let SPAConfig::DiricDirichlet(info) = self {
            Ok(info)
        } else {
            bail!("Expected Dirac-Dirichlet SPA Info.")
        }
    }

    pub fn alphabet_size(&self) -> u32 {
        match self {
            SPAConfig::Dirichlet(info) => info.alphabet_size,
            SPAConfig::LZ78(info) => info.inner_config.alphabet_size(),
            SPAConfig::Discrete(_) => 2,
            SPAConfig::DiricDirichlet(_) => 2,
        }
    }

    pub fn compute_training_loss(&self) -> bool {
        match self {
            SPAConfig::Dirichlet(config) => config.training_log_loss,
            SPAConfig::LZ78(config) => config.inner_config.compute_training_loss(),
            SPAConfig::Discrete(_) => true,
            SPAConfig::DiricDirichlet(_) => true,
        }
    }

    pub fn maybe_get_gamma(&self) -> Option<f64> {
        match self {
            SPAConfig::Dirichlet(info) => Some(info.gamma),
            _ => None,
        }
    }

    pub fn maybe_set_gamma(&mut self, gamma: f64) {
        match self {
            SPAConfig::Dirichlet(info) => info.gamma = gamma,
            _ => {}
        }
    }

    pub fn get_new_state(&self) -> SPAState {
        SPAState::get_new_state(self)
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteThetaConfig {
    pub theta_pmf: Array1<f64>,
    pub theta_values: Array1<f64>,
}

impl ToFromBytes for DiscreteThetaConfig {
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

impl DiscreteThetaConfig {
    pub fn new(theta_pmf: &[f64], theta_values: &[f64]) -> Self {
        Self {
            theta_pmf: Array1::from_vec(theta_pmf.to_vec()),
            theta_values: Array1::from_vec(theta_values.to_vec()),
        }
    }

    pub fn new_enum(theta_pmf: &[f64], theta_values: &[f64]) -> SPAConfig {
        SPAConfig::Discrete(Self::new(theta_pmf, theta_values))
    }
}

#[derive(Debug, Clone)]
pub struct DiracDirichletConfig {
    pub dirichlet_config: Box<SPAConfig>,
    pub disc_config: DiscreteThetaConfig,
    pub dirichlet_weight: f64,
}

impl ToFromBytes for DiracDirichletConfig {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.dirichlet_config.to_bytes()?;
        bytes.extend(self.disc_config.to_bytes()?);
        bytes.put_f64_le(self.dirichlet_weight);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let dirichlet_config = Box::new(SPAConfig::from_bytes(bytes)?);
        let disc_config = DiscreteThetaConfig::from_bytes(bytes)?;
        let dirichlet_weight = bytes.get_f64_le();
        Ok(Self {
            dirichlet_config,
            disc_config,
            dirichlet_weight,
        })
    }
}

impl DiracDirichletConfig {
    pub fn new(theta_pmf: &[f64], theta_values: &[f64], gamma: f64, dirichlet_weight: f64) -> Self {
        Self {
            dirichlet_config: Box::new(SPAConfig::Dirichlet(DirichletConfig {
                gamma,
                alphabet_size: 2,
                lb_and_temp: LbAndTemp::Skip,
                training_log_loss: true,
            })),
            disc_config: DiscreteThetaConfig::new(theta_pmf, theta_values),
            dirichlet_weight,
        }
    }
    pub fn new_enum(
        theta_pmf: &[f64],
        theta_values: &[f64],
        gamma: f64,
        dirichlet_weight: f64,
    ) -> SPAConfig {
        SPAConfig::DiricDirichlet(Self::new(theta_pmf, theta_values, gamma, dirichlet_weight))
    }
}

#[derive(Debug, Clone)]
pub struct DirichletConfig {
    pub gamma: f64,
    pub alphabet_size: u32,
    pub lb_and_temp: LbAndTemp,
    training_log_loss: bool,
}

impl ToFromBytes for DirichletConfig {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_f64_le(self.gamma);
        bytes.put_u32_le(self.alphabet_size);
        bytes.extend(self.lb_and_temp.to_bytes()?);
        bytes.put_u8(self.training_log_loss as u8);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let gamma = bytes.get_f64_le();
        let alphabet_size = bytes.get_u32_le();
        let lb_and_temp = LbAndTemp::from_bytes(bytes)?;
        let training_log_loss = bytes.get_u8() > 0;
        Ok(Self {
            gamma,
            alphabet_size,
            lb_and_temp,
            training_log_loss,
        })
    }
}

pub struct DirichletConfigBuilder {
    gamma: f64,
    alphabet_size: u32,
    lb_and_temp: LbAndTemp,
    training_log_loss: bool,
}

impl DirichletConfigBuilder {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            alphabet_size,
            gamma: 0.5,
            lb_and_temp: LbAndTemp::Skip,
            training_log_loss: true,
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

    pub fn compute_training_log_loss(&mut self, training_log_loss: bool) -> &mut Self {
        self.training_log_loss = training_log_loss;
        self
    }

    pub fn build(&self) -> DirichletConfig {
        DirichletConfig {
            alphabet_size: self.alphabet_size,
            gamma: self.gamma,
            lb_and_temp: self.lb_and_temp,
            training_log_loss: self.training_log_loss,
        }
    }

    pub fn build_enum(&self) -> SPAConfig {
        SPAConfig::Dirichlet(self.build())
    }
}

#[derive(Debug, Clone)]
pub struct LZ78Config {
    pub inner_config: Box<SPAConfig>,
    pub adaptive_gamma: AdaptiveGamma,
    pub ensemble: Ensemble,
    pub backshift_parsing: BackshiftParsing,
    pub freeze_tree: bool,
    pub track_parents: bool,
    pub max_depth: Option<u32>,
}

impl ToFromBytes for LZ78Config {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.inner_config.to_bytes()?;
        bytes.extend(self.adaptive_gamma.to_bytes()?);
        bytes.extend(self.ensemble.to_bytes()?);
        bytes.extend(self.backshift_parsing.to_bytes()?);
        bytes.put_u8(self.freeze_tree as u8);
        bytes.put_u8(self.track_parents as u8);
        match self.max_depth {
            Some(d) => {
                bytes.put_u8(0);
                bytes.put_u32_le(d)
            }
            None => {
                bytes.put_u8(1);
            }
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let inner_config = Box::new(SPAConfig::from_bytes(bytes)?);
        let adaptive_gamma = AdaptiveGamma::from_bytes(bytes)?;
        let ensemble = Ensemble::from_bytes(bytes)?;
        let backshift_parsing = BackshiftParsing::from_bytes(bytes)?;
        let freeze_tree = bytes.get_u8() > 0;
        let track_parents = bytes.get_u8() > 0;
        let max_depth = if bytes.get_u8() == 0 {
            Some(bytes.get_u32_le())
        } else {
            None
        };

        Ok(Self {
            inner_config,
            adaptive_gamma,
            ensemble,
            backshift_parsing,
            freeze_tree,
            track_parents,
            max_depth,
        })
    }
}

pub struct LZ78ConfigBuilder {
    inner_config: Box<SPAConfig>,
    adaptive_gamma: AdaptiveGamma,
    ensemble: Ensemble,
    backshift_parsing: BackshiftParsing,
    track_parents: bool,
    max_depth: Option<u32>,
}

impl LZ78ConfigBuilder {
    pub fn new(inner_info: SPAConfig) -> Self {
        Self {
            inner_config: Box::new(inner_info),
            adaptive_gamma: AdaptiveGamma::None,
            ensemble: Ensemble::None,
            backshift_parsing: BackshiftParsing::Disabled,
            track_parents: true,
            max_depth: None,
        }
    }

    pub fn adaptive_gamma(mut self, adaptive_gamma: AdaptiveGamma) -> Self {
        self.adaptive_gamma = adaptive_gamma;
        self
    }

    pub fn ensemble(mut self, ensemble: Ensemble) -> Self {
        self.ensemble = ensemble;
        self
    }

    pub fn track_parents(mut self, track_parents: bool) -> Self {
        self.track_parents = track_parents;
        self
    }

    pub fn max_depth(mut self, max_depth: Option<u32>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn backshift(mut self, desired_context_length: u64, break_at_phrase: bool) -> Self {
        self.backshift_parsing = BackshiftParsing::Enabled {
            desired_context_length,
            break_at_phrase,
        };
        self
    }

    pub fn build(self) -> LZ78Config {
        LZ78Config {
            inner_config: self.inner_config,
            adaptive_gamma: self.adaptive_gamma,
            ensemble: self.ensemble,
            backshift_parsing: self.backshift_parsing,
            freeze_tree: false,
            track_parents: self.track_parents,
            max_depth: self.max_depth,
        }
    }

    pub fn build_enum(self) -> SPAConfig {
        SPAConfig::LZ78(self.build())
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
        break_at_phrase: bool,
    },
    Disabled,
}

impl BackshiftParsing {
    /// Returns a tuple of (desired_ctx_len, min_spa_training_pts), or zeros
    /// if backshift parsing is disabled
    pub fn get_config(&self) -> (u64, bool) {
        match self {
            BackshiftParsing::Enabled {
                desired_context_length,
                break_at_phrase,
            } => (*desired_context_length, *break_at_phrase),
            BackshiftParsing::Disabled => (0, false),
        }
    }
}

impl ToFromBytes for BackshiftParsing {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            BackshiftParsing::Enabled {
                desired_context_length,
                break_at_phrase,
            } => {
                bytes.put_u8(0);
                bytes.put_u64_le(*desired_context_length);
                bytes.put_u8(*break_at_phrase as u8);
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
                let break_at_phrase = bytes.get_u8() > 0;
                Self::Enabled {
                    desired_context_length,
                    break_at_phrase,
                }
            }
            1 => Self::Disabled,
            _ => bail!("unexpected backshift parsing type"),
        })
    }
}
