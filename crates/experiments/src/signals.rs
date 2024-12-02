use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

pub fn triangle_pulse(height: u8, step: u8) -> Vec<u8> {
    let mut result: Vec<u8> = Vec::with_capacity(height as usize * 2 - 1);
    for i in (0..height).step_by(step as usize) {
        result.push(i);
    }
    for i in (0..(height - 1)).rev().step_by(step as usize) {
        result.push(i);
    }

    return result;
}

pub fn add_noise(signal: &Vec<u8>, noise_range: u8, max_val: u8) -> Vec<u8> {
    let rng_samples = Uniform::new(-(noise_range as i16), noise_range as i16 + 1)
        .sample_iter(&mut thread_rng())
        .take(signal.len())
        .collect_vec();

    signal
        .iter()
        .zip(rng_samples)
        .map(|(&x, noise)| ((x as i16) + noise).max(0).min(max_val as i16) as u8)
        .collect_vec()
}
