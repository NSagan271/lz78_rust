use std::time::Instant;

use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Datasets, TrainCli},
    utils::{
        default_character_map, read_c4_realnewslike, read_fashion_mnist, read_file_to_string,
        read_tinystories, read_wikitext, DatasetPartition,
    },
};

fn shakespeare_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let data = read_file_to_string(&format!("{}/shakespeare/input.txt", cli.data_dir.clone()))?;

    let character_map = CharacterMap::from_data(&data);
    character_map.save_to_file(cli.save_path.clone() + ".charmap")?;

    let mut spa = LZ78SPA::new(character_map.alphabet_size, cli.gamma);

    let tic = Instant::now();
    let seq = CharacterSequence::from_data_filtered(data, character_map.clone());
    spa.train_on_block(&seq, false)?;
    spa.save_to_file(cli.save_path.clone())?;

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    Ok(spa)
}

fn tinystories_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let data = read_tinystories(&format!("{}/TinyStories", cli.data_dir))?;

    let character_map = default_character_map();
    character_map.save_to_file(cli.save_path.clone() + ".charmap")?;

    let mut spa = LZ78SPA::new(character_map.alphabet_size, cli.gamma);

    let tic = Instant::now();
    let n = data.len() / 4;
    for s in data.into_iter().take(n) {
        let seq = CharacterSequence::from_data_filtered(s, character_map.clone());
        spa.train_on_block(&seq, false)?;
    }
    spa.save_to_file(cli.save_path.clone())?;

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    Ok(spa)
}

fn c4_realnewslike_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let character_map = default_character_map();
    character_map.save_to_file(cli.save_path.clone() + ".charmap")?;

    let mut spa = LZ78SPA::new(character_map.alphabet_size, cli.gamma);

    let tic = Instant::now();
    for i in 0..10 {
        println!("Part {i}");
        let text = read_c4_realnewslike(&format!("{}/c4", cli.data_dir.clone()), i as u64)?;

        for sample in text {
            let seq = CharacterSequence::from_data_filtered(sample, character_map.clone());
            spa.train_on_block(&seq, !cli.start_at_root)?;
        }

        spa.save_to_file(cli.save_path.clone())?;
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    Ok(spa)
}

fn wikitext_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let character_map = default_character_map();
    let mut text = read_wikitext(&format!("{}/wikitext", cli.data_dir))?;
    if let Some(samples) = cli.samples {
        text = text.into_iter().take(samples as usize).collect_vec();
    }

    let mut spa = LZ78SPA::new(character_map.alphabet_size, cli.gamma);

    let n_loops = cli.repeat;
    let tic = Instant::now();
    for _ in 0..n_loops {
        for sample in text.iter() {
            let seq = CharacterSequence::from_data_filtered(sample.clone(), character_map.clone());
            spa.train_on_block(&seq, !cli.start_at_root)?;
        }
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    Ok(spa)
}

fn fashion_mnist_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
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
    let mut spa = LZ78SPA::new(alpha_size, cli.gamma);

    let n_loops = cli.repeat;
    let tic = Instant::now();
    for _ in 0..n_loops {
        for img in bytes.iter() {
            let seq = U8Sequence::from_data(img.clone(), alpha_size)?;
            spa.train_on_block(&seq, !cli.start_at_root)?;
        }
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
        spa.get_normalized_log_loss()
    );

    Ok(spa)
}

fn main() {
    let cli = TrainCli::parse();
    let save_path = cli.save_path.clone();

    let spa = match cli.dataset {
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

    spa.save_to_file(save_path).expect("write failed");
}
