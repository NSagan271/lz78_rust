use anyhow::Result;
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

use crate::{
    sequence::Sequence,
    util::{apply_temp_and_topk_to_spa, sample_from_pdf},
};

use super::{states::SPAState, SPAParams, SPA};

pub trait GenerationSPA: SPA {
    /// Called when "seeding" the text generation
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
        gen_params: &GenerationParams,
        gen_state: &mut SPAState,
        context_syms: &[u32],
    ) -> Result<(u32, f64)>;
}

pub struct GenerationParams {
    pub temperature: f64,
    pub top_k: u32,
}

impl GenerationParams {
    pub fn default() -> Self {
        Self {
            temperature: 0.5,
            top_k: 10,
        }
    }

    pub fn new(temperature: f64, top_k: u32) -> Self {
        Self { temperature, top_k }
    }
}

pub fn gen_symbol_from_spa(
    rng_sample: f64,
    gen_params: &GenerationParams,
    spa: &[f64],
) -> Result<(u32, f64)> {
    let orig_spa = spa;
    let mut spa = spa.to_vec();

    apply_temp_and_topk_to_spa(
        &mut spa,
        gen_params.temperature,
        Some(gen_params.top_k as usize),
    );

    let new_sym = sample_from_pdf(&spa, rng_sample) as u32;
    let loss = -orig_spa[new_sym as usize].log2();
    Ok((new_sym, loss))
}

pub fn generate_sequence<S, T>(
    spa: &mut S,
    n: u64,
    spa_params: &mut SPAParams,
    gen_params: &GenerationParams,
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
            gen_params,
            &mut gen_state,
            &output_syms,
        )?;
        output_sequence.put_sym(sym)?;
        output_syms.push(sym);
        loss += new_loss;
    }

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use crate::{
        sequence::{CharacterSequence, Sequence, SequenceParams},
        spa::{
            basic_spas::DirichletSPA,
            generation::{generate_sequence, GenerationParams},
            lz_transform::LZ78SPA,
            AdaptiveGamma, BackshiftParsing, Ensemble, SPAParams, SPA,
        },
    };

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let mut params = SPAParams::new_lz78_dirichlet(
            input.alphabet_size(),
            0.5,
            AdaptiveGamma::None,
            Ensemble::Average(5),
            BackshiftParsing::Enabled {
                desired_context_length: 5,
                min_spa_training_points: 1,
            },
            false,
        );
        let mut state = params.get_new_state();
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &mut params, &mut state)
            .expect("failed to train spa");

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

        generate_sequence(
            &mut spa,
            100,
            &mut params,
            &GenerationParams::new(0.0, 10),
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
            &GenerationParams::new(1.0, 1),
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
            &GenerationParams::new(2.0, 5),
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
            &GenerationParams::new(0.5, 5),
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
