use std::{
    fs::{remove_file, File},
    time::Instant,
};

use clap::Parser;
use itertools::Itertools;
use log::warn;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, U8Sequence},
    spa::{lz78_spa_monte_carlo_branch_lengths, DirichletSPA, SPAParams, LZ78SPA, SPA},
    storage::ToFromBytes,
};
use lz78_experiments::{
    argparse::{Datasets, TrainCli},
    utils::{
        default_character_map, read_c4_realnewslike, read_fashion_mnist, read_file_to_string,
        read_tinystories, read_wikitext, DatasetPartition,
    },
};
use plotpy::{Histogram, Plot};

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

fn text_experiment(
    input: impl Iterator<Item = String>,
    mut character_map: CharacterMap,
    cli: TrainCli,
) -> anyhow::Result<()> {
    character_map.add('~'); // "character not found in map" placeholder
    character_map.save_to_file(cli.save_path.clone() + ".charmap")?;

    let params = SPAParams::new_lz78_dirichlet(character_map.alphabet_size, cli.gamma, true);
    let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;

    let tic = Instant::now();

    let mut bytes_processed = 0;

    for s in input {
        let s = character_map.filter_string_and_replace(&s, '~');
        bytes_processed += s.as_bytes().len();

        let seq = CharacterSequence::from_data_filtered(s, character_map.clone());
        spa.reset_state();
        spa.train_on_block(&seq, &params)?;
    }
    let time = tic.elapsed().as_secs_f32();

    let debug = spa.get_debug_info();
    let leaf_depths = debug.leaf_depths.values().map(|x| *x).collect_vec();

    let max_depth = debug.max_depth;
    let mean_depth =
        leaf_depths.iter().map(|&x| x as u64).sum::<u64>() as f64 / leaf_depths.len() as f64;

    leaf_depth_hist(
        &format!(
            "plots/leaf_depth_hist_{}",
            cli.save_path.clone().replace("/", "_").replace(".", "_")
        ),
        leaf_depths,
    );

    println!(
        "Trained on {:.2} MiB with log loss {:.2} in {time:.3} seconds",
        bytes_processed as f64 / 1024. / 1024.,
        spa.get_normalized_log_loss(),
    );
    println!("Max leaf depth: {max_depth}, mean leaf depth: {mean_depth:.2}");

    let longest_branch = character_map.try_decode_all(debug.get_longest_branch())?;
    println!("Longest branch: \n--------------------\n{longest_branch}\n--------------------");

    let mc_n = 500;
    let monte_carlo_leaf_depths = lz78_spa_monte_carlo_branch_lengths(&mut spa, mc_n)?;
    let mean_depth_mc = monte_carlo_leaf_depths
        .iter()
        .map(|&x| x as u64)
        .sum::<u64>() as f64
        / monte_carlo_leaf_depths.len() as f64;
    println!("Mean leaf depth (monte carlo w/ N={mc_n}): {mean_depth_mc:.2}");
    leaf_depth_hist(
        &format!(
            "plots/mc_leaf_depth_hist_{}",
            cli.save_path.clone().replace("/", "_").replace(".", "_")
        ),
        monte_carlo_leaf_depths,
    );

    spa.clear_debug_info();
    spa.save_to_file(cli.save_path.clone())?;

    let f = File::open(cli.save_path)?;
    let file_len = f.metadata()?.len() as f64 / 1024. / 1024.;
    println!("Saved SPA size: {file_len:.2} MiB");

    Ok(())
}

fn shakespeare_experiment(cli: TrainCli) -> anyhow::Result<()> {
    let data = read_file_to_string(&format!("{}/shakespeare/input.txt", cli.data_dir.clone()))?;
    let character_map = CharacterMap::from_data(&data);

    text_experiment(vec![data].into_iter(), character_map, cli)
}

fn tinystories_experiment(cli: TrainCli) -> anyhow::Result<()> {
    let data = read_tinystories(&format!("{}/TinyStories", cli.data_dir))?;
    let character_map = default_character_map();
    let n = data.len() / 4;

    text_experiment(data.into_iter().take(n), character_map, cli)
}

fn c4_realnewslike_experiment(cli: TrainCli) -> anyhow::Result<()> {
    let character_map = default_character_map();
    let data_dir = cli.data_dir.clone();
    let iter = (0..10)
        .map(|i| {
            read_c4_realnewslike(&format!("{}/c4", data_dir.clone()), i as u64)
                .unwrap_or(vec![])
                .into_iter()
        })
        .flatten();
    text_experiment(iter, character_map, cli)
}

fn wikitext_experiment(cli: TrainCli) -> anyhow::Result<()> {
    let character_map = default_character_map();
    let mut text = read_wikitext(&format!("{}/wikitext", cli.data_dir))?;
    if let Some(samples) = cli.samples {
        text = text.into_iter().take(samples as usize).collect_vec();
    }

    text_experiment(text.into_iter(), character_map, cli)
}

fn fashion_mnist_experiment(cli: TrainCli) -> anyhow::Result<()> {
    let (mut bytes, _) = read_fashion_mnist(
        &format!("{}/fashion_mnist", cli.data_dir),
        DatasetPartition::Train,
    )?;
    bytes = bytes
        .into_iter()
        .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
        .collect_vec();
    if let Some(samples) = cli.samples {
        bytes = bytes.into_iter().take(samples as usize).collect_vec();
    }

    let alpha_size = 256 / 32;
    let params = SPAParams::new_lz78_dirichlet(alpha_size, cli.gamma, true);
    let mut spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;

    let n_loops = cli.repeat;
    let tic = Instant::now();
    for _ in 0..n_loops {
        for img in bytes.iter() {
            let seq = U8Sequence::from_data(img.clone(), alpha_size)?;
            spa.train_on_block(&seq, &params)?;
            spa.reset_state();
        }
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    spa.save_to_file(cli.save_path).expect("write failed");
    Ok(())
}

fn main() {
    stderrlog::new()
        .module(module_path!())
        .verbosity(log::Level::Info)
        .init()
        .unwrap();
    let cli = TrainCli::parse();
    match cli.dataset {
        Datasets::Wikitext => wikitext_experiment(cli).expect("wikitext experiment failed"),
        Datasets::FashionMnist => {
            fashion_mnist_experiment(cli).expect("fashion mnist experiment failed")
        }
        Datasets::C4 => c4_realnewslike_experiment(cli).expect("c4 experiment failed"),
        Datasets::Shakespeare => {
            shakespeare_experiment(cli).expect("Shakespeare experiment failed")
        }
        Datasets::TinyStories => {
            tinystories_experiment(cli).expect("tinystories experiment failed")
        }
        _ => {
            println!("Training not available for the dataset provided! Exiting.");
            return;
        }
    };
}
