use crate::{sequence::Sequence, util::sample_from_pdf};

use super::{params::SPAParams, states::SPAState, util::apply_temp_and_topk_to_spa, SPATree, SPA};
use anyhow::Result;
use itertools::Itertools;
use ndarray::Array1;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

pub trait GenerationSPATree: SPATree {
    fn input_seed_data_symbol(
        &self,
        idx: u64,
        sym: u32,
        params: &mut SPAParams,
        gen_state: &mut SPAState,
    ) -> Result<f64>;

    /// Generates one symbol and updates the state of the SPA accordingly.
    fn generate_one_symbol(
        &self,
        idx: u64,
        rng_sample: f64,
        params: &mut SPAParams,
        gen_state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)>;
}

pub trait GenerationSPA: SPA {
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &mut SPAParams,
        gen_state: &mut SPAState,
    ) -> Result<f64>;

    /// Generates one symbol and updates the state of the SPA accordingly.
    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &mut SPAParams,
        gen_state: &mut SPAState,
        context_syms: &[u32],
        temperature: f64,
        topk: Option<u32>,
    ) -> Result<(u32, f64)>;
}

pub fn generate_sequence<S, T>(
    spa: &mut S,
    n: u64,
    spa_params: &mut SPAParams,
    temperature: f64,
    topk: Option<u32>,
    seed_data: Option<&T>,
    output_sequence: &mut T,
) -> Result<f64>
where
    S: GenerationSPA,
    T: Sequence,
{
    let mut loss = 0.0;
    let mut gen_state = spa_params.get_new_state();
    let mut output_syms = Vec::with_capacity(n as usize);

    if let Some(data) = seed_data {
        output_syms.reserve_exact(data.len() as usize);
        for sym in data.iter() {
            loss += spa.input_seed_data_symbol(sym, spa_params, &mut gen_state)?;
            output_syms.push(sym);
        }
    }
    let rng_samples = Uniform::new(0.0, 1.0)
        .sample_iter(&mut thread_rng())
        .take(n as usize)
        .collect_vec();

    for i in 0..n {
        let (sym, new_loss) = spa.generate_one_symbol(
            rng_samples[i as usize],
            spa_params,
            &mut gen_state,
            &output_syms,
            temperature,
            topk,
        )?;
        output_sequence.put_sym(sym)?;
        output_syms.push(sym);
        loss += new_loss;
    }

    Ok(loss)
}

pub fn gen_symbol_from_spa(
    rng_sample: f64,
    spa: &Array1<f64>,
    temperature: f64,
    topk: Option<u32>,
) -> Result<(u32, f64)> {
    let mut new_spa = spa.clone();
    apply_temp_and_topk_to_spa(&mut new_spa, temperature, topk);
    let new_sym = sample_from_pdf(&new_spa, rng_sample) as u32;
    let loss = -spa[new_sym as usize].log2();
    Ok((new_sym, loss))
}

#[cfg(test)]
mod tests {
    use crate::{
        sequence::{CharacterSequence, Sequence, SequenceParams},
        spa::{
            dirichlet::DirichletSPATree,
            generation::generate_sequence,
            lz_transform::LZ78SPA,
            params::{DirichletParamsBuilder, LZ78ParamsBuilder},
            SPA,
        },
    };

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(100),
        );
        let mut params =
            LZ78ParamsBuilder::new(DirichletParamsBuilder::new(input.alphabet_size()).build_enum())
                .backshift(4, 1, true)
                .build_enum();
        let mut state = params.get_new_state();

        let mut spa: LZ78SPA<DirichletSPATree> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

        generate_sequence(
            &mut spa,
            100,
            &mut params,
            0.0,
            None,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            &mut generation_output,
        )
        .expect("generating data failed");

        println!(
            "Temperature 0, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output2 =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

        generate_sequence(
            &mut spa,
            100,
            &mut params,
            1.0,
            Some(1),
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            &mut generation_output2,
        )
        .expect("generating data failed");

        println!(
            "Temperature 1, topk 1, seed \"hello\": {:?}",
            generation_output2.data
        );

        assert_eq!(generation_output.data, generation_output2.data);

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

        generate_sequence(
            &mut spa,
            100,
            &mut params,
            1.0,
            None,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            &mut generation_output,
        )
        .expect("generating data failed");

        println!(
            "Temperature 2, topk 5, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();
        generate_sequence(
            &mut spa,
            100,
            &mut params,
            0.5,
            Some(10),
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
            &mut generation_output,
        )
        .expect("generating data failed");

        println!(
            "Temperature 0.5, topk 10, seed \"hello\": {:?}",
            generation_output.data
        );
    }
}
