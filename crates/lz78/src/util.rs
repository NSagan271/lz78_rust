use itertools::Itertools;

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