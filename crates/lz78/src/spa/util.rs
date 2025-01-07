use bytes::{Buf, BufMut};
use itertools::Itertools;

use super::AdaptiveGamma;
use crate::storage::ToFromBytes;
use anyhow::{bail, Result};

pub fn apply_temp_and_topk_to_spa(spa: &mut [f64], temp: f64, k: Option<usize>) {
    let most_likely_next_sym = (0..spa.len() as u32)
        .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
        .unwrap() as usize;

    let k = k.unwrap_or(spa.len());

    // If temperature is 0.0, we just compute the argmax of the SPA.
    if temp == 0.0 {
        for i in 0..spa.len() {
            spa[i] = 0.0;
        }
        spa[most_likely_next_sym] = 1.0;
        return;
    }

    if temp != 1.0 {
        for i in 0..spa.len() {
            spa[i] = 2.0_f64.powf(spa[i].log2() / temp);
        }
    }

    // top-k sampling
    (0..spa.len())
        .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
        .take(spa.len() - k)
        .for_each(|i| {
            spa[i as usize] = 0.0;
        });
    let sum: f64 = spa.iter().sum();
    for i in 0..spa.len() {
        spa[i] /= sum;
    }
}

pub fn apply_lb_to_spa(spa: &mut [f64], lb: f64) {
    for i in 0..spa.len() {
        spa[i] = spa[i].max(lb);
    }
    let sum: f64 = spa.iter().sum();
    for i in 0..spa.len() {
        spa[i] /= sum;
    }
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

pub fn apply_lb_and_temp_to_spa(spa: &mut [f64], lb_temp_params: LbAndTemp) {
    match lb_temp_params {
        LbAndTemp::TempFirst { lb, temp } => {
            apply_temp_and_topk_to_spa(spa, temp, None);
            apply_lb_to_spa(spa, lb);
        }
        LbAndTemp::LbFirst { lb, temp } => {
            apply_lb_to_spa(spa, lb);
            apply_temp_and_topk_to_spa(spa, temp, None);
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
