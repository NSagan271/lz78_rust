use std::time::Instant;

use anyhow::{bail, Result};
use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterSequence, Sequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Datasets, ImageClassificationCli},
    utils::{
        default_character_map, quantize_images, read_cifar10, read_fashion_mnist, read_imdb,
        read_mnist, read_spam, DatasetPartition,
    },
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

fn num_classes(dataset: Datasets) -> u64 {
    if dataset == Datasets::Imdb || dataset == Datasets::Spam {
        2
    } else {
        10
    }
}

fn quantize_and_map_to_boxed_sequence(
    data: Vec<Vec<u8>>,
    quant_strength: u8,
) -> Result<Vec<Box<dyn Sequence>>> {
    let data = quantize_images(data, quant_strength);
    let alpha_size = 256 / quant_strength as u32;
    let n = data.len();
    let seqs = data
        .into_iter()
        .filter_map(|v| {
            if let Ok(x) = U8Sequence::from_data(v, alpha_size) {
                let val: Box<dyn Sequence> = Box::new(x);
                Some(val)
            } else {
                None
            }
        })
        .collect_vec();
    if seqs.len() != n {
        bail!("Error making some sequences");
    }
    Ok(seqs)
}

/// Returns a tuple of (datasets, class)
fn get_data(
    dataset: Datasets,
    data_dir: String,
    partition: DatasetPartition,
    quantization_strength: Option<u8>,
) -> Result<(Vec<Box<dyn Sequence>>, Vec<u8>)> {
    match dataset {
        lz78_experiments::argparse::Datasets::FashionMnist => {
            let (data, classes) =
                read_fashion_mnist(&format!("{data_dir}/fashion_mnist"), partition)?;
            Ok((
                quantize_and_map_to_boxed_sequence(data, quantization_strength.unwrap_or(1))?,
                classes,
            ))
        }
        lz78_experiments::argparse::Datasets::Mnist => {
            let (data, classes) = read_mnist(&format!("{data_dir}/mnist"), partition)?;
            Ok((
                quantize_and_map_to_boxed_sequence(data, quantization_strength.unwrap_or(1))?,
                classes,
            ))
        }
        lz78_experiments::argparse::Datasets::Cifar10 => {
            let (data, classes) = read_cifar10(&format!("{data_dir}/cifar"), partition)?;
            Ok((
                quantize_and_map_to_boxed_sequence(data, quantization_strength.unwrap_or(1))?,
                classes,
            ))
        }
        lz78_experiments::argparse::Datasets::Imdb => {
            let (data, classes) = read_imdb(&format!("{data_dir}/imdb"), partition)?;
            let charmap = default_character_map();
            let data = data
                .into_iter()
                .map(|s| {
                    let seq: Box<dyn Sequence> =
                        Box::new(CharacterSequence::from_data_filtered(s, charmap.clone()));
                    seq
                })
                .collect_vec();
            Ok((data, classes))
        }
        lz78_experiments::argparse::Datasets::Spam => {
            let (data, classes) = read_spam(&format!("{data_dir}/enron_spam_data"), partition)?;
            let charmap = default_character_map();
            let data = data
                .into_iter()
                .map(|s| {
                    let seq: Box<dyn Sequence> =
                        Box::new(CharacterSequence::from_data_filtered(s, charmap.clone()));
                    seq
                })
                .collect_vec();
            Ok((data, classes))
        }
        _ => bail!("dataset not available for classification"),
    }
}

fn main() {
    let cli = ImageClassificationCli::parse();

    let (sequences, classes) = get_data(
        cli.dataset,
        cli.data_dir.clone(),
        DatasetPartition::Train,
        Some(cli.quant_strength),
    )
    .expect("error reading dataset");
    let alpha_size = sequences[0].alphabet_size(); // all sequences have the same alphabet size

    let n_class = num_classes(cli.dataset);

    let tic = Instant::now();
    let class_to_seqs = classes.into_iter().zip(sequences).into_group_map();

    let mut spas = (0..n_class)
        .into_par_iter()
        .map(|class| {
            let mut spa = LZ78SPA::new(alpha_size, cli.gamma);
            for seq in class_to_seqs.get(&(class as u8)).unwrap() {
                for _ in 0..cli.repeat {
                    spa.train_on_block(seq.as_ref(), false)
                        .expect("train failed");
                }
            }
            spa
        })
        .collect::<Vec<_>>();

    let time = tic.elapsed().as_secs_f32();
    println!("Trained SPA in {time:.3} seconds");

    let (test, classes) = get_data(
        cli.dataset,
        cli.data_dir.clone(),
        DatasetPartition::Test,
        Some(cli.quant_strength),
    )
    .expect("error reading dataset");

    let mut correct = 0;
    let n = test.len();
    for (seq, true_class) in test.into_iter().zip(classes) {
        let class = spas
            .par_iter_mut()
            .enumerate()
            .map(|(i, spa)| {
                (
                    i,
                    spa.compute_test_loss(seq.as_ref(), false)
                        .expect("failed to compute test loss"),
                )
            })
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap()
            .0;

        if class == true_class as usize {
            correct += 1;
        }
    }

    println!("Accuracy: {}", correct as f32 / n as f32);
}
