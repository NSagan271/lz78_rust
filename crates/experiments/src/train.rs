use std::{
    fs::{remove_file, File},
    io::Write,
    time::Instant,
};

use anyhow::{bail, Result};
use clap::Parser;
use itertools::Itertools;
use log::warn;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, SequenceParams},
    spa::{
        basic_spas::DirichletSPA,
        causally_processed::{
            CausalProcessor, CausallyProcessedLZ78SPA, CausallyProcessedLZ78SPAParams,
            ManualQuantizer,
        },
        ctw::CTW,
        lz_transform::{lz78_spa_monte_carlo_branch_lengths, LZ78SPA},
        LZ78SPAParams, SPAParams, SPA,
    },
    storage::ToFromBytes,
};
use lz78_experiments::argparse::SPATypes as ArgparseSpaTypes;
use lz78_experiments::{
    argparse::{Datasets, TrainCli},
    data::{read_c4_realnewslike, read_file_to_string, read_tinystories, read_wikitext},
    spa_ablation_utils::SPATypes,
    utils::{default_char_quantizer, default_character_map},
};
use plotpy::{Histogram, Plot};
use serde::{Deserialize, Serialize};

fn leaf_depth_hist(plot_path: &str, leaf_depths: Vec<u32>) {
    let mut histogram = Histogram::new();
    histogram.draw(&vec![leaf_depths], &vec!["Leaf Depths"]);

    let mut plot = Plot::new();
    plot.add(&histogram)
        .set_frame_border(true, false, true, false)
        .grid_labels_legend("values", "count");

    if let Err(e) = plot.save(&plot_path) {
        warn!("Possible plotting error: {e}");
    }

    //clean up after pyplot
    if let Err(e) = remove_file(format!("{plot_path}.py")) {
        warn!("Could not cleanup pyplot files: {e}");
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Losses {
    avg_losses: Vec<f64>,
    ns: Vec<u64>,
}

fn train_spa<T: SPA>(
    input: impl Iterator<Item = String>,
    character_map: &CharacterMap,
    params: &SPAParams,
    spa: &mut T,
    cli: TrainCli,
) -> Result<()> {
    let tic = Instant::now();

    let mut avg_losses = Vec::new();
    let mut ns = Vec::new();

    let mut bytes_processed = 0;
    let mut loss = 0.0;
    for s in input {
        let s = character_map.filter_string_and_replace(&s, '~');
        bytes_processed += s.as_bytes().len();

        let seq = CharacterSequence::from_data_filtered(s, character_map.clone());
        spa.reset_state();
        loss += spa.train_on_block(&seq, &params)?;

        avg_losses.push(loss / spa.num_symbols_seen() as f64);
        ns.push(spa.num_symbols_seen());
    }
    let time = tic.elapsed().as_secs_f32();

    println!(
        "Trained on {:.2} MiB with log loss {:.2} in {time:.3} seconds",
        bytes_processed as f64 / 1024. / 1024.,
        loss / spa.num_symbols_seen() as f64,
    );

    let serialized_losses =
        serde_pickle::to_vec(&Losses { avg_losses, ns }, Default::default()).unwrap();

    let mut file = File::create(format!("{}_losses.pkl", cli.save_path))?;
    file.write_all(&serialized_losses)?;

    Ok(())
}

fn text_experiment_internal_debug<T: SPA>(
    input: impl Iterator<Item = String>,
    character_map: &CharacterMap,
    params: &SPAParams,
    spa: &mut LZ78SPA<T>,
    cli: TrainCli,
) -> Result<()> {
    let path = cli.save_path.clone();
    train_spa(input, character_map, params, spa, cli)?;

    let debug = spa.get_debug_info();
    let leaf_depths = debug.leaf_depths.values().map(|x| *x).collect_vec();

    let max_depth = debug.max_depth;
    let mean_depth =
        leaf_depths.iter().map(|&x| x as u64).sum::<u64>() as f64 / leaf_depths.len() as f64;

    leaf_depth_hist(
        &format!(
            "plots/leaf_depth_hist_{}",
            path.clone().replace("/", "_").replace(".", "_")
        ),
        leaf_depths,
    );
    println!("Max leaf depth: {max_depth}, mean leaf depth: {mean_depth:.2}");

    let longest_branch = character_map.try_decode_all(debug.get_longest_branch())?;
    println!("Longest branch: \n--------------------\n{longest_branch}\n--------------------");

    let mc_n = 500;
    let monte_carlo_leaf_depths = lz78_spa_monte_carlo_branch_lengths(spa, mc_n)?;
    let mean_depth_mc = monte_carlo_leaf_depths
        .iter()
        .map(|&x| x as u64)
        .sum::<u64>() as f64
        / monte_carlo_leaf_depths.len() as f64;
    println!("Mean leaf depth (monte carlo w/ N={mc_n}): {mean_depth_mc:.2}");
    leaf_depth_hist(
        &format!(
            "plots/mc_leaf_depth_hist_{}",
            path.clone().replace("/", "_").replace(".", "_")
        ),
        monte_carlo_leaf_depths,
    );

    Ok(())
}

fn text_experiment_internal<T: SPA>(
    input: impl Iterator<Item = String>,
    character_map: &CharacterMap,
    params: &SPAParams,
    spa: &mut T,
    cli: TrainCli,
) -> Result<()> {
    train_spa(input, character_map, params, spa, cli)?;

    Ok(())
}

fn text_experiment_internal_quatized<T: SPA>(
    input: impl Iterator<Item = String>,
    character_map: &CharacterMap,
    params: &mut LZ78SPAParams,
    quantizer: &ManualQuantizer<CharacterSequence>,
) -> Result<(CausallyProcessedLZ78SPAParams, CausallyProcessedLZ78SPA<T>)> {
    let params = CausallyProcessedLZ78SPAParams::new(
        quantizer.orig_params.alphabet_size(),
        quantizer.quant_params.alphabet_size(),
        params.inner_params.as_ref().clone(),
        false,
    );
    let mut spa: CausallyProcessedLZ78SPA<T> = CausallyProcessedLZ78SPA::new(&params)?;
    let tic = Instant::now();

    let mut bytes_processed = 0;
    let mut losses = Vec::new();
    let mut ns = Vec::new();

    for s in input {
        let s = character_map.filter_string_and_replace(&s, '~');
        bytes_processed += s.as_bytes().len();

        let seq = quantizer.get_causally_processed_seq(CharacterSequence::from_data_filtered(
            s,
            character_map.clone(),
        ))?;
        spa.reset_state();
        spa.train_on_block(&seq, &params)?;

        losses.push(spa.get_normalized_log_loss());
        ns.push(spa.num_symbols_seen());
    }
    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained on {:.2} MiB with log loss {:.2} in {time:.3} seconds",
        bytes_processed as f64 / 1024. / 1024.,
        spa.get_normalized_log_loss(),
    );

    // println!("losses = {losses:?}");
    // println!("n = {ns:?}");

    Ok((params, spa))
}

fn text_experiment(
    input: impl Iterator<Item = String>,
    character_map: &CharacterMap,
    cli: TrainCli,
) -> anyhow::Result<()> {
    let seq_params = SequenceParams::CharMap(character_map.clone());
    let path = cli.save_path.clone();
    match cli.spa_type {
        ArgparseSpaTypes::Dirichlet => {
            if cli.debug {
                bail!("Debug and quantizaiton not supported for Dirichlet SPA");
            }
            let params = SPAParams::new_dirichlet(character_map.alphabet_size, cli.gamma);
            let mut spa = DirichletSPA::new(&params)?;
            text_experiment_internal(input, character_map, &params, &mut spa, cli)?;
            SPATypes::Dirichlet(spa, params, seq_params).save_to_file(path.clone())?;
        }
        ArgparseSpaTypes::LZ78Dirichlet => {
            let params =
                SPAParams::new_lz78_dirichlet(character_map.alphabet_size, cli.gamma, cli.debug);
            let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;
            if cli.debug {
                text_experiment_internal_debug(input, character_map, &params, &mut spa, cli)?;
            } else {
                text_experiment_internal(input, character_map, &params, &mut spa, cli)?;
            }
            SPATypes::LZ78Dirichlet(spa, params, seq_params).save_to_file(path.clone())?;
        }
        ArgparseSpaTypes::LZ78CTW => {
            let params = SPAParams::new_lz78_ctw(
                character_map.alphabet_size,
                cli.gamma,
                cli.ctw_depth,
                cli.debug,
            );
            let mut spa: LZ78SPA<CTW> = LZ78SPA::new(&params)?;
            if cli.debug {
                text_experiment_internal_debug(input, character_map, &params, &mut spa, cli)?;
            } else {
                text_experiment_internal(input, character_map, &params, &mut spa, cli)?;
            }
            return SPATypes::LZ78CTW(spa, params, seq_params).save_to_file(path.clone());
        }
        ArgparseSpaTypes::CharQuantizedLZ78 => {
            if cli.debug {
                bail!("Debug not implemented for quantized SPA");
            }

            let mut params =
                LZ78SPAParams::new_dirichlet(character_map.alphabet_size, cli.gamma, false);
            let quantizer = default_char_quantizer()?;
            let (params, spa) =
                text_experiment_internal_quatized(input, character_map, &mut params, &quantizer)?;
            SPATypes::CharQuantizedLZ78(spa, quantizer, params).save_to_file(path.clone())?;
        }
    }

    let f = File::open(path)?;
    let file_len = f.metadata()?.len() as f64 / 1024. / 1024.;
    println!("Saved SPA size: {file_len:.2} MiB");
    Ok(())
}

fn main() -> Result<()> {
    stderrlog::new()
        .module(module_path!())
        .verbosity(log::Level::Info)
        .init()
        .unwrap();
    let cli = TrainCli::parse();
    match cli.dataset {
        Datasets::Wikitext => {
            let character_map = default_character_map();
            let mut text = read_wikitext(&format!("{}/wikitext", cli.data_dir))?;
            if let Some(samples) = cli.samples {
                text = text.into_iter().take(samples as usize).collect_vec();
            }

            text_experiment(text.into_iter(), &character_map, cli)?
        }
        Datasets::C4 => {
            let character_map = default_character_map();
            let data_dir = cli.data_dir.clone();
            let iter = (0..1)
                .map(|i| {
                    read_c4_realnewslike(&format!("{}/c4", data_dir.clone()), i as u64)
                        .unwrap_or(vec![])
                        .into_iter()
                })
                .flatten();
            text_experiment(iter, &character_map, cli)?;
        }
        Datasets::TinyStories => {
            let data = read_tinystories(&format!("{}/TinyStories", cli.data_dir))?;
            let character_map = default_character_map();
            let n = data.len() / 4;

            text_experiment(data.into_iter().take(n), &character_map, cli)?;
        }
        Datasets::Shakespeare => {
            let data =
                read_file_to_string(&format!("{}/shakespeare/input.txt", cli.data_dir.clone()))?;
            let character_map = CharacterMap::from_data(&data);

            text_experiment(vec![data].into_iter(), &character_map, cli)?;
        }
        _ => {
            bail!("Training not available for the dataset provided! Exiting.");
        }
    };

    Ok(())
}

// fn fashion_mnist_experiment(cli: TrainCli) -> anyhow::Result<()> {
//     let (mut bytes, _) = read_fashion_mnist(
//         &format!("{}/fashion_mnist", cli.data_dir),
//         DatasetPartition::Train,
//     )?;
//     bytes = bytes
//         .into_iter()
//         .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
//         .collect_vec();
//     if let Some(samples) = cli.samples {
//         bytes = bytes.into_iter().take(samples as usize).collect_vec();
//     }

//     let alpha_size = 256 / 32;
//     let params = SPAParams::new_lz78_dirichlet(alpha_size, cli.gamma, true);
//     let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;

//     let n_loops = cli.repeat;
//     let tic = Instant::now();
//     for _ in 0..n_loops {
//         for img in bytes.iter() {
//             let seq = U8Sequence::from_data(img.clone(), alpha_size)?;
//             spa.train_on_block(&seq, &params)?;
//             spa.reset_state();
//         }
//     }

//     let time = tic.elapsed().as_secs_f32();
//     println!(
//         "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
//         spa.get_normalized_log_loss()
//     );

//     spa.save_to_file(cli.save_path)?;
//     Ok(())
// }
