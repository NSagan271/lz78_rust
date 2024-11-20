use anyhow::Result;
use clap::Parser;
use lz78::{
    sequence::{CharacterMap, CharacterSequence},
    spa::{
        basic_spas::DirichletSPA,
        generation::{generate_sequence, GenerationParams},
        lz_transform::LZ78SPA,
        SPAParams,
    },
    storage::ToFromBytes,
};
use lz78_experiments::argparse::{Datasets, GenerateCli};

fn text_gen_experiment(cli: GenerateCli, mut spa: LZ78SPA<DirichletSPA>) -> Result<()> {
    let character_map = CharacterMap::from_file(cli.save_path + ".charmap")?;

    let mut generate_output = CharacterSequence::new(character_map.clone());
    let seed_data = cli.seed_data.unwrap_or("".to_string());

    let gen_params = GenerationParams::new(cli.temperature, cli.topk, cli.min_context, 2);

    generate_sequence(
        &mut spa,
        cli.n,
        &SPAParams::default_lz78_dirichlet(character_map.alphabet_size),
        &gen_params,
        Some(&CharacterSequence::from_data_filtered(
            seed_data.clone(),
            character_map.clone(),
        )),
        &mut generate_output,
    )?;

    println!("{seed_data}{}", generate_output.data);

    Ok(())
}

// fn fashion_mnist_experiment(
//     _cli: GenerateCli,
//     mut spa: LZ78SPA<LZ78SPA<DirichletSPA>>,
// ) -> Result<()> {
//     let mut generate_output = U8Sequence::new(256);

//     let (mut test_set, _) = read_fashion_mnist("data/fashion_mnist", DatasetPartition::Test)?;
//     test_set = test_set
//         .into_iter()
//         .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
//         .collect_vec();
//     let test_img = U8Sequence::from_data(test_set[0][0..28 * 28 / 2].to_vec(), 256 / 32)?;
//     spa.generate_data(
//         &mut generate_output,
//         28 * 28 / 2,
//         1000,
//         0.1,
//         3,
//         Some(&test_img),
//         None,
//     )?;

//     println!("test_data = np.array({:?}).reshape(-1, 28)", test_img.data);
//     println!(
//         "gen_data = np.array({:?}).reshape(-1, 28)",
//         generate_output.data
//     );

//     Ok(())
// }

fn main() {
    let cli = GenerateCli::parse();
    let spa: LZ78SPA<DirichletSPA> =
        LZ78SPA::from_file(cli.save_path.clone()).expect("read spa failed");

    match cli.dataset {
        // Datasets::FashionMnist => {
        //     fashion_mnist_experiment(cli, spa).expect("fashion mnist experiment failed")
        // }
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
