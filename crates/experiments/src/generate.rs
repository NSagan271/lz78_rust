use anyhow::{bail, Result};
use clap::Parser;
use lz78::{spa::generation::GenerationParams, storage::ToFromBytes};
use lz78_experiments::{
    argparse::{Datasets, GenerateCli},
    spa_ablation_utils::SPATypes,
};

fn text_gen_experiment(cli: GenerateCli) -> Result<()> {
    let mut spa = SPATypes::from_file(cli.save_path)?;
    let gen_params = GenerationParams::new(cli.temperature, cli.topk);
    spa.generate_string(cli.n, &gen_params, &cli.seed_data)?;
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

fn main() -> anyhow::Result<()> {
    let cli = GenerateCli::parse();

    match cli.dataset {
        // Datasets::FashionMnist => fashion_mnist_experiment(cli, spa)?,
        Datasets::Wikitext => text_gen_experiment(cli)?,
        Datasets::C4 => text_gen_experiment(cli)?,
        Datasets::Shakespeare => text_gen_experiment(cli)?,
        Datasets::TinyStories => text_gen_experiment(cli)?,
        _ => {
            bail!("Sequence generation not available for provided dataset");
        }
    }

    Ok(())
}
