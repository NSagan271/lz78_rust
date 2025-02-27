use std::collections::HashMap;

use pyo3::pyfunction;

#[pyfunction]
pub fn mu_k(seq: Vec<u32>, alpha_size: u32, k: u32) -> f64 {
    let mut k_plus_one_counts: HashMap<u64, u64> =
        HashMap::with_capacity(alpha_size.pow(k + 1) as usize);
    let mut k_counts: HashMap<u64, u64> = HashMap::with_capacity(alpha_size.pow(k) as usize);

    let mut k_enc: u64 = 0;
    let mut k_plus_one_enc: u64 = 0;

    // loop through data
    for i in 0..seq.len() {
        if k > 0 {
            k_enc *= alpha_size as u64;
            k_enc %= (alpha_size as u64).pow(k);
            k_enc += seq[i] as u64;
        }

        k_plus_one_enc *= alpha_size as u64;
        k_plus_one_enc %= (alpha_size as u64).pow(k + 1);
        k_plus_one_enc += seq[i] as u64;

        if i + 1 >= k as usize {
            k_counts.insert(k_enc, k_counts.get(&k_enc).unwrap_or(&0) + 1);
        }

        if i >= k as usize {
            k_plus_one_counts.insert(
                k_plus_one_enc,
                k_plus_one_counts.get(&k_plus_one_enc).unwrap_or(&0) + 1,
            );
        }
    }

    // compute empirical entropy
    let mut emp_entropy = 0f64;
    for (val, count) in k_plus_one_counts {
        let parent_count = k_counts[&(val / alpha_size as u64)];
        emp_entropy -=
            (count as f64 / seq.len() as f64) * (count as f64 / parent_count as f64).log2();
    }

    emp_entropy
}

#[cfg(test)]
mod tests {
    use crate::markov::mu_k;

    #[test]
    fn test_zero_order_entropy() {
        assert_eq!(
            mu_k(
                vec![0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                2,
                0
            ),
            1.0
        );
        assert_eq!(
            mu_k(
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1
                ],
                2,
                0
            ),
            1.0
        );
    }

    #[test]
    fn sanity_check() {
        mu_k(
            vec![
                0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1,
                1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1,
            ],
            2,
            5,
        );
    }
}
