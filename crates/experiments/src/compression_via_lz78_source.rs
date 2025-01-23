use std::io::{stdout, Write};

use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use lz78::{
    encoder::{Encoder, LZ8Encoder},
    sequence::U32Sequence,
    source::{DiscreteBinaryThetaSPA, LZ78Source},
    spa::{
        basic_spas::DirichletSPA, util::LbAndTemp, AdaptiveGamma, BackshiftParsing, Ensemble,
        SPAParams,
    },
};
use lz78_experiments::argparse::SourceCompressionCli;
use ndarray::arr1;
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Only binary experiments so far
const A: u32 = 2;

// fn empirical_entropy(data: &[usize]) -> f64 {
//     let n = data.len();
//     let mut counts: HashMap<usize, u64> = HashMap::new();
//     for &val in data.iter() {
//         counts.insert(val, *counts.get(&val).unwrap_or(&0) + 1);
//     }

//     let mut h = 0.;
//     for (_, count) in counts {
//         let p = count as f64 / n as f64;
//         h += p * (1. / p).log2();
//     }

//     h
// }

fn generate_from_lz78_source_parallel(n_thread: u64, k: u64, gamma: f64) -> Result<Vec<Vec<u32>>> {
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(n_thread as usize);

    for res in (0..n_thread)
        .into_par_iter()
        .map(|_| {
            let mut params = SPAParams::new_lz78_dirichlet(
                A,
                gamma,
                LbAndTemp::Skip,
                AdaptiveGamma::None,
                Ensemble::None,
                BackshiftParsing::Disabled,
                false,
            );
            let mut state = params.get_new_state();
            let mut source: LZ78Source<DirichletSPA> = LZ78Source::new(&params, &mut thread_rng())?;
            source.generate_symbols(k, &mut thread_rng(), &mut params, &mut state)
        })
        .collect::<Vec<_>>()
    {
        result.push(res?.data);
    }

    Ok(result)
}

fn get_entropy_rate(cli: &SourceCompressionCli) -> f64 {
    match cli.data_generator {
        lz78_experiments::argparse::DataGenerators::Bernoulli => {
            if cli.prob_one == 0. || cli.prob_one == 1. {
                0.
            } else {
                cli.prob_one * (1.0 / cli.prob_one).log2()
                    + (1.0 - cli.prob_one) * (1.0 / (1.0 - cli.prob_one)).log2()
            }
        }
        lz78_experiments::argparse::DataGenerators::BernoulliLZ78Source => 0.,
    }
}

fn get_sequence_to_compress(cli: &SourceCompressionCli) -> Result<Vec<u32>> {
    match cli.data_generator {
        lz78_experiments::argparse::DataGenerators::Bernoulli => Ok((0..cli.k_max)
            .map(|_| {
                if thread_rng().gen::<f64>() < cli.prob_one {
                    1u32
                } else {
                    0
                }
            })
            .collect_vec()),
        lz78_experiments::argparse::DataGenerators::BernoulliLZ78Source => {
            let mut params = SPAParams::new_discrete(arr1(&[0.5, 0.5]), arr1(&[0., 1.]));
            let mut state = params.get_new_state();
            let mut ber_src: LZ78Source<DiscreteBinaryThetaSPA> =
                LZ78Source::new(&params, &mut thread_rng())?;
            Ok(ber_src
                .generate_symbols(cli.k_max, &mut thread_rng(), &mut params, &mut state)?
                .data)
        }
    }
}

fn main() -> Result<()> {
    let cli = SourceCompressionCli::parse();

    let h2 = get_entropy_rate(&cli);
    eprintln!("h2(q) = {h2}",);

    for k in 1..=cli.k_max {
        print!("k={}{}", k, if k == cli.k_max { "\n" } else { ", " });
    }
    for _trial in 0..cli.trials {
        let rand_seq = get_sequence_to_compress(&cli)?;
        // Find the LZ78 compression ratio
        let encoded = LZ8Encoder::new().encode(&U32Sequence::from_data(rand_seq.clone(), A)?)?;
        eprintln!("LZ78 Compression Ratio: {}", encoded.compression_ratio());

        let mut n_ks: Vec<u64> = Vec::with_capacity(cli.k_max as usize);

        let mut curr_nk = 0;
        let mut curr_k = 1;
        let mut curr_results =
            generate_from_lz78_source_parallel(cli.max_thread, cli.k_max, cli.gamma)?;
        while (n_ks.len() as u64) < cli.k_max {
            if rand_seq[0..curr_k] == curr_results[(curr_nk % cli.max_thread) as usize][0..curr_k] {
                let log_n_k_by_k = if curr_nk == 0 {
                    0.
                } else {
                    (curr_nk as f32 + 1.).log2() / curr_k as f32
                };
                print!(
                    "{log_n_k_by_k}{}",
                    if (n_ks.len() as u64) == cli.k_max - 1 {
                        "\n"
                    } else {
                        ", "
                    }
                );
                let _ = stdout().flush();
                n_ks.push(curr_nk);
                curr_k += 1;
            } else {
                curr_nk += 1;
                if curr_nk % cli.max_thread == 0 {
                    curr_results =
                        generate_from_lz78_source_parallel(cli.max_thread, cli.k_max, cli.gamma)?;
                }
            }
        }
    }
    Ok(())
}
