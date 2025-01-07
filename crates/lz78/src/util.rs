use itertools::Itertools;

use crate::spa::AdaptiveGamma;

/// Given a PDF and a sample of a Uniform([0, 1)) random variable, randomly
/// sample a number from 0 to pdf.len() - 1 as per the PDF.
pub fn sample_from_pdf(pdf: &[f64], sample: f64) -> u64 {
    let cdf = pdf
        .iter()
        .scan(0.0_f64, |sum, i| {
            *sum += i;
            Some(*sum)
        })
        .collect_vec();
    (0..pdf.len())
        .filter(|i| sample <= cdf[*i as usize])
        .next()
        .unwrap_or(pdf.len() - 1) as u64
}

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

#[derive(Debug, Clone, Copy)]
pub enum LbOrTemp {
    TempFirst,
    LbFirst,
    Skip,
}

pub fn apply_lb_and_temp_to_spa(spa: &mut [f64], lb: f64, temp: f64, lb_or_temp: LbOrTemp) {
    match lb_or_temp {
        LbOrTemp::TempFirst => {
            apply_temp_and_topk_to_spa(spa, temp, None);
            apply_lb_to_spa(spa, lb);
        }
        LbOrTemp::LbFirst => {
            apply_lb_to_spa(spa, lb);
            apply_temp_and_topk_to_spa(spa, temp, None);
        }
        LbOrTemp::Skip => {}
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
