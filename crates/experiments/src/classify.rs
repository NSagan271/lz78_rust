use core::f64;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterSequence, Sequence, U8Sequence},
    spa::{
        basic_spas::DirichletSPA, lz_transform::LZ78SPA, util::LbAndTemp, AdaptiveGamma,
        BackshiftParsing, Ensemble, SPAParams, SPA,
    },
};
use lz78_experiments::data::{read_cifar10, read_fashion_mnist, read_imdb, read_mnist, read_spam};
use lz78_experiments::{
    argparse::{Datasets, ImageClassificationCli},
    utils::{default_character_map, quantize_images, DatasetPartition},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

fn num_classes(dataset: Datasets) -> u8 {
    if dataset == Datasets::Imdb || dataset == Datasets::Spam {
        2
    } else {
        10
    }
}

fn train_spa_parallel<T>(
    sequences: Vec<T>,
    classes: Vec<u8>,
    params: &SPAParams,
    cli: &ImageClassificationCli,
) -> Result<Vec<LZ78SPA<DirichletSPA>>>
where
    T: Sequence,
{
    let tic = Instant::now();
    let n_class = num_classes(cli.dataset);

    let class_to_seqs = classes.into_iter().zip(sequences).into_group_map();
    let mut spas = Vec::with_capacity(n_class as usize);

    for res in (0..n_class)
        .into_par_iter()
        .map(|class| -> Result<LZ78SPA<DirichletSPA>> {
            let mut spa = LZ78SPA::new(&params)?;
            let mut state = params.get_new_state();
            for seq in class_to_seqs.get(&class).unwrap() {
                let mut new_params = params.clone();
                for _ in 0..cli.repeat {
                    spa.train_on_block(seq, &mut new_params, &mut state)
                        .expect("train failed");
                }
            }
            Ok(spa)
        })
        .collect::<Vec<_>>()
    {
        spas.push(res?);
    }
    let time = tic.elapsed().as_secs_f32();
    println!("Trained SPA in {time:.3} seconds");

    Ok(spas)
}

fn test_spa_parallel<T>(
    test: Vec<T>,
    classes: Vec<u8>,
    spas: &mut [LZ78SPA<DirichletSPA>],
    params: &SPAParams,
) -> Result<f32>
where
    T: Sequence,
{
    let mut correct = 0;
    let n = test.len();
    for (seq, true_class) in test.into_iter().zip(classes) {
        let class = spas
            .par_iter_mut()
            .enumerate()
            .map(|(i, spa)| {
                let mut new_params = params.clone();
                (
                    i,
                    spa.test_on_block(&seq, &mut new_params, &mut params.get_new_state())
                        .unwrap_or(f64::INFINITY),
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
    Ok(correct as f32 / n as f32)
}

fn quantize_images_and_make_sequences(
    data: Vec<Vec<u8>>,
    quant_bits: u8,
) -> Result<Vec<U8Sequence>> {
    if quant_bits > 8 {
        bail!("quant_bits cannot be > 8 for an 8-bit image")
    }
    let data = quantize_images(data, quant_bits);
    let alpha_size = 2u32.pow(quant_bits as u32);
    let n = data.len();
    let seqs = data
        .into_iter()
        .filter_map(|v| {
            if let Ok(x) = U8Sequence::from_data(v, alpha_size) {
                Some(x)
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

fn image_experiment(
    train_data: Vec<Vec<u8>>,
    train_classes: Vec<u8>,
    test_data: Vec<Vec<u8>>,
    test_classes: Vec<u8>,
    cli: &ImageClassificationCli,
) -> Result<()> {
    let train = quantize_images_and_make_sequences(train_data, cli.quant_bits)?;
    let params = SPAParams::new_lz78_dirichlet(
        train[0].alphabet_size(),
        cli.gamma,
        LbAndTemp::Skip,
        AdaptiveGamma::None,
        Ensemble::None,
        BackshiftParsing::Disabled,
        false,
    );

    let mut spas = train_spa_parallel(train, train_classes, &params, &cli)?;

    let test = quantize_images_and_make_sequences(test_data, cli.quant_bits)?;
    let _accuracy = test_spa_parallel(test, test_classes, &mut spas, &params)?;

    Ok(())
}

fn text_experiment(
    train_data: Vec<String>,
    train_classes: Vec<u8>,
    test_data: Vec<String>,
    test_classes: Vec<u8>,
    cli: &ImageClassificationCli,
) -> Result<()> {
    let charmap = default_character_map();
    let train = train_data
        .into_iter()
        .map(|s| CharacterSequence::from_data_filtered(s, charmap.clone()))
        .collect_vec();

    let params = SPAParams::new_lz78_dirichlet(
        train[0].alphabet_size(),
        cli.gamma,
        LbAndTemp::Skip,
        AdaptiveGamma::None,
        Ensemble::None,
        BackshiftParsing::Disabled,
        false,
    );
    let mut spas = train_spa_parallel(train, train_classes, &params, &cli)?;

    let test = test_data
        .into_iter()
        .map(|s| CharacterSequence::from_data_filtered(s, charmap.clone()))
        .collect_vec();
    let _accuracy = test_spa_parallel(test, test_classes, &mut spas, &params)?;

    Ok(())
}

fn main() -> Result<()> {
    let cli = ImageClassificationCli::parse();
    let data_dir = cli.data_dir.clone();

    match cli.dataset {
        lz78_experiments::argparse::Datasets::FashionMnist => {
            let dir = format!("{data_dir}/fashion_mnist");
            let (train_data, train_classes) = read_fashion_mnist(&dir, DatasetPartition::Train)?;
            let (test_data, test_classes) = read_fashion_mnist(&dir, DatasetPartition::Test)?;
            image_experiment(train_data, train_classes, test_data, test_classes, &cli)?;
        }
        lz78_experiments::argparse::Datasets::Mnist => {
            let dir = format!("{data_dir}/mnist");
            let (train_data, train_classes) = read_mnist(&dir, DatasetPartition::Train)?;
            let (test_data, test_classes) = read_mnist(&dir, DatasetPartition::Test)?;
            image_experiment(train_data, train_classes, test_data, test_classes, &cli)?;
        }
        lz78_experiments::argparse::Datasets::Cifar10 => {
            let dir = format!("{data_dir}/cifar");
            let (train_data, train_classes) = read_cifar10(&dir, DatasetPartition::Train)?;
            let (test_data, test_classes) = read_cifar10(&dir, DatasetPartition::Test)?;
            image_experiment(train_data, train_classes, test_data, test_classes, &cli)?;
        }
        lz78_experiments::argparse::Datasets::Imdb => {
            let dir = format!("{data_dir}/imdb");
            let (train_data, train_classes) = read_imdb(&dir, DatasetPartition::Train)?;
            let (test_data, test_classes) = read_imdb(&dir, DatasetPartition::Train)?;
            text_experiment(train_data, train_classes, test_data, test_classes, &cli)?;
        }
        lz78_experiments::argparse::Datasets::Spam => {
            let dir = format!("{data_dir}/enron_spam_data");
            let (train_data, train_classes) = read_spam(&dir, DatasetPartition::Train)?;
            let (test_data, test_classes) = read_spam(&dir, DatasetPartition::Train)?;
            text_experiment(train_data, train_classes, test_data, test_classes, &cli)?;
        }
        _ => bail!("dataset not available for classification"),
    }

    Ok(())
}
