use super::AdaptiveGamma;
use crate::storage::ToFromBytes;
use anyhow::{bail, Result};
use bytes::{Buf, BufMut};
use itertools::Itertools;
use ndarray::Array1;
use ndarray_stats::QuantileExt;

pub fn apply_temp_and_topk_to_spa(spa: &mut Array1<f64>, temp: f64, k: Option<u32>) {
    let most_likely_next_sym = spa.argmax().unwrap();
    let k = k.unwrap_or(spa.len() as u32) as usize;

    // If temperature is 0.0, we just compute the argmax of the SPA.
    if temp == 0.0 {
        spa.fill(0.0);
        spa[most_likely_next_sym] = 1.0;
        return;
    }
    // top-k sampling
    let top_k_elem = *spa
        .iter()
        .sorted_by(|x, y| y.total_cmp(&x))
        .skip(k)
        .next()
        .unwrap_or(&-1.0);

    spa.map_mut(|x| *x = if *x >= top_k_elem { *x } else { 0.0 });

    if temp != 1.0 {
        *spa = (spa.clone().log2() / temp).exp2();
    }

    *spa /= spa.sum();
}

pub fn apply_lb_to_spa(spa: &mut Array1<f64>, lb: f64) {
    spa.map_mut(|x| *x = x.max(lb));
    *spa /= spa.sum();
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LbAndTemp {
    TempFirst { lb: f64, temp: f64 },
    LbFirst { lb: f64, temp: f64 },
    Skip,
}

impl LbAndTemp {
    pub fn lb_only(lb: f64) -> Self {
        Self::LbFirst { lb, temp: 1.0 }
    }

    pub fn temp_only(temp: f64) -> Self {
        Self::TempFirst { lb: 0.0, temp }
    }

    /// Returns a tuple of (lb, temp)
    pub fn get_vals(&self) -> (f64, f64) {
        match self {
            LbAndTemp::TempFirst { lb, temp } => (*lb, *temp),
            LbAndTemp::LbFirst { lb, temp } => (*lb, *temp),
            LbAndTemp::Skip => (0.0, 1.0),
        }
    }

    pub fn set_vals(&mut self, new_lb: f64, new_temp: f64) -> Result<()> {
        match self {
            LbAndTemp::TempFirst { lb, temp } => {
                *lb = new_lb;
                *temp = new_temp
            }
            LbAndTemp::LbFirst { lb, temp } => {
                *lb = new_lb;
                *temp = new_temp
            }
            LbAndTemp::Skip => {
                if new_lb != 0.0 || new_temp != 1.0 {
                    bail!("Tried to set lower bound or temperature, but \"lb_or_temp_first\" is \"SKIP\". Set \"lb_or_temp_first\" first.")
                }
            }
        }

        Ok(())
    }
}

impl ToFromBytes for LbAndTemp {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            LbAndTemp::TempFirst { lb, temp } => {
                bytes.put_u8(0);
                bytes.put_f64_le(*lb);
                bytes.put_f64_le(*temp);
            }
            LbAndTemp::LbFirst { lb, temp } => {
                bytes.put_u8(1);
                bytes.put_f64_le(*lb);
                bytes.put_f64_le(*temp);
            }
            LbAndTemp::Skip => {
                bytes.put_u8(2);
            }
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let tpe = bytes.get_u8();
        if tpe > 2 {
            bail!("unexpected LBAndTemp type");
        }
        if tpe == 2 {
            return Ok(Self::Skip);
        }
        let lb = bytes.get_f64_le();
        let temp = bytes.get_f64_le();
        if tpe == 0 {
            Ok(Self::TempFirst { lb, temp })
        } else {
            Ok(Self::LbFirst { lb, temp })
        }
    }
}

pub fn apply_lb_and_temp_to_spa(
    spa: &mut Array1<f64>,
    lb_temp_params: LbAndTemp,
    topk: Option<u32>,
) {
    match lb_temp_params {
        LbAndTemp::TempFirst { lb, temp } => {
            apply_temp_and_topk_to_spa(spa, temp, topk);
            apply_lb_to_spa(spa, lb);
        }
        LbAndTemp::LbFirst { lb, temp } => {
            apply_lb_to_spa(spa, lb);
            apply_temp_and_topk_to_spa(spa, temp, topk);
        }
        LbAndTemp::Skip => {}
    }
}

pub fn adaptive_gamma(
    old_gamma: f64,
    adaptive_gamma: AdaptiveGamma,
    depth: u32,
    parent_count: u64,
) -> f64 {
    match adaptive_gamma {
        AdaptiveGamma::Inverse => old_gamma / (1 + depth) as f64,
        AdaptiveGamma::Count => old_gamma / parent_count as f64,
        AdaptiveGamma::None => old_gamma,
    }
}
