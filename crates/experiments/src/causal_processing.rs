use std::time::Instant;

use anyhow::Result;
use lz78::{
    sequence::U8Sequence,
    spa::{
        basic_spas::DirichletSPA,
        causally_processed::{
            CausalProcessor, CausallyProcessedLZ78SPA, CausallyProcessedLZ78SPAParams,
            IntegerScalarQuantizer,
        },
        lz_transform::LZ78SPA,
        SPAParams, SPA,
    },
};
use lz78_experiments::{
    data::read_fasta,
    signals::{add_noise, triangle_pulse},
    utils::{get_codon_processor, nucleotides_to_codon_seqs},
};

fn genetics() -> Result<()> {
    const GAMMA: f64 = 1.;

    let exons = read_fasta("data/genes/dab1_rna.fna")?;

    let params = SPAParams::new_lz78_dirichlet(65, GAMMA, false);
    let mut regular_spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;

    let quantizer = get_codon_processor();
    let proc_params =
        CausallyProcessedLZ78SPAParams::new_dirichlet(65, quantizer.alphabet_size(), GAMMA, false);
    let mut proc_spa: CausallyProcessedLZ78SPA<DirichletSPA> =
        CausallyProcessedLZ78SPA::new(&proc_params)?;

    let mut original_losses: Vec<f64> = Vec::new();
    let mut proc_losses: Vec<f64> = Vec::new();
    let mut n_so_far: Vec<u64> = Vec::new();

    for exon in exons {
        let codon_seq = nucleotides_to_codon_seqs(&exon)?;
        for codons in codon_seq {
            let processed = quantizer.get_causally_processed_seq(codons.clone())?;

            regular_spa.train_on_block(&codons, &params)?;
            proc_spa.train_on_block(&processed, &proc_params)?;
            regular_spa.reset_state();
            proc_spa.reset_state();

            original_losses.push(regular_spa.get_normalized_log_loss());
            proc_losses.push(proc_spa.get_normalized_log_loss());
            n_so_far.push(regular_spa.num_symbols_seen());
        }
    }

    println!("Losses of original SPA: {original_losses:?}");
    println!("Losses of processed SPA: {proc_losses:?}");
    println!("n: {n_so_far:?}");

    Ok(())
}

/// Generate a deterministic sequence and then add noise, and see if the
/// causally-processed LZ78 SPA does a better job.
fn synthetic_data_experiment() -> Result<()> {
    const GAMMA: f64 = 0.1;
    const BLOCK_SIZE: usize = 1000;
    const REPEAT: usize = 100_000;

    let data = triangle_pulse(20, 2).repeat(REPEAT);
    let noisy = add_noise(&data, 1, 20);

    let params = SPAParams::new_lz78_dirichlet(20, GAMMA, false);
    let mut regular_spa: LZ78SPA<DirichletSPA> = LZ78SPA::new(&params)?;

    let mut original_losses: Vec<f64> = Vec::new();

    let tic = Instant::now();
    for i in 0..(data.len() / BLOCK_SIZE) {
        regular_spa.train_on_block(
            &U8Sequence::from_data(noisy[(BLOCK_SIZE * i)..(BLOCK_SIZE * (i + 1))].to_vec(), 21)?,
            &params,
        )?;
        original_losses.push(regular_spa.get_normalized_log_loss());
    }
    let orig_time = tic.elapsed().as_secs_f32();

    println!("Losses of original SPA: {original_losses:?}");

    let quantizer: IntegerScalarQuantizer<U8Sequence> = IntegerScalarQuantizer::new(20, 2);
    let proc_params =
        CausallyProcessedLZ78SPAParams::new_dirichlet(21, quantizer.alphabet_size(), GAMMA, false);
    let mut proc_spa: CausallyProcessedLZ78SPA<DirichletSPA> =
        CausallyProcessedLZ78SPA::new(&proc_params)?;

    let mut proc_losses: Vec<f64> = Vec::new();

    let tic = Instant::now();
    for i in 0..(data.len() / BLOCK_SIZE) {
        proc_spa.train_on_block(
            &quantizer.get_causally_processed_seq(U8Sequence::from_data(
                noisy[(BLOCK_SIZE * i)..(BLOCK_SIZE * (i + 1))].to_vec(),
                21,
            )?)?,
            &proc_params,
        )?;
        proc_losses.push(proc_spa.get_normalized_log_loss());
    }
    let proc_time = tic.elapsed().as_secs_f32();

    println!("Losses of processed SPA: {proc_losses:?}");

    eprintln!(
        "Time of original SPA: {orig_time:.3}, time of processing + processed SPA: {proc_time:.3}"
    );

    Ok(())
}

fn main() -> Result<()> {
    // synthetic_data_experiment()
    genetics()
}
