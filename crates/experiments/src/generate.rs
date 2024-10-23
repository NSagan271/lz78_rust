use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Datasets, GenerateCli},
    utils::{read_fashion_mnist, DatasetPartition},
};

fn text_gen_experiment(cli: GenerateCli, mut spa: LZ78SPA) -> Result<()> {
    let character_map = CharacterMap::from_file(cli.save_path + ".charmap")?;

    let mut generate_output = CharacterSequence::new(character_map.clone());
    let seed_data = cli.seed_data.unwrap_or("".to_string());
    spa.generate_data(
        &mut generate_output,
        cli.n,
        cli.min_context,
        cli.temperature,
        cli.topk,
        Some(&CharacterSequence::from_data_filtered(
            seed_data.clone(),
            character_map.clone(),
        )),
    )?;

    println!("{seed_data}{}", generate_output.data);

    Ok(())
}

fn fashion_mnist_experiment(_cli: GenerateCli, mut spa: LZ78SPA) -> Result<()> {
    let mut generate_output = U8Sequence::new(256);

    let (mut test_set, _) = read_fashion_mnist("data/fashion_mnist", DatasetPartition::Test)?;
    test_set = test_set
        .into_iter()
        .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
        .collect_vec();
    let test_img = U8Sequence::from_data(test_set[0][0..28 * 28 / 2].to_vec(), 256 / 32)?;
    spa.generate_data(
        &mut generate_output,
        28 * 28 / 2,
        1000,
        0.1,
        3,
        Some(&test_img),
    )?;

    println!("test_data = np.array({:?}).reshape(-1, 28)", test_img.data);
    println!(
        "gen_data = np.array({:?}).reshape(-1, 28)",
        generate_output.data
    );

    Ok(())
}

fn main() {
    let cli = GenerateCli::parse();
    let spa = LZ78SPA::from_file(cli.save_path.clone()).expect("read spa failed");

    match cli.dataset {
        Datasets::FashionMnist => {
            fashion_mnist_experiment(cli, spa).expect("fashion mnist experiment failed")
        }
        Datasets::Wikitext => text_gen_experiment(cli, spa).expect("wikitext experiment failed"),
        Datasets::C4 => text_gen_experiment(cli, spa).expect("c4 experiment failed"),
        Datasets::Shakespeare => {
            text_gen_experiment(cli, spa).expect("Shakespeare experiment failed")
        }
        Datasets::TinyStories => {
            text_gen_experiment(cli, spa).expect("tinystories experiment failed")
        }
        _ => {
            println!("Sequence generation not available for provided dataset");
            return;
        }
    }
}
