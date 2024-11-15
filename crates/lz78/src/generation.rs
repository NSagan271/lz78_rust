use anyhow::Result;
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

use crate::{
    sequence::Sequence,
    spa::{SPAParams, SPA},
};

pub trait GenerationSPA: SPA {
    /// Called at the end of sequence generation.
    fn cleanup_post_generation(&mut self);

    /// Called when "seeding" the text generation
    fn input_seed_data_symbol(&mut self, sym: u32, params: &SPAParams) -> Result<f64>;

    /// Generates one symbol and updates the state of the SPA accordingly.
    fn generate_one_symbol(
        &mut self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &GenerationParams,
    ) -> Result<(u32, f64)>;
}

pub struct GenerationParams {
    pub temperature: f64,
    pub top_k: u32,
    pub desired_context_length: u64,
    pub min_spa_training_points: u64,
}

impl GenerationParams {
    pub fn default() -> Self {
        Self {
            desired_context_length: 10,
            min_spa_training_points: 2,
            temperature: 0.5,
            top_k: 10,
        }
    }

    pub fn new(
        temperature: f64,
        top_k: u32,
        desired_context_length: u64,
        min_spa_training_points: u64,
    ) -> Self {
        Self {
            temperature,
            top_k,
            desired_context_length,
            min_spa_training_points,
        }
    }
}

pub fn generate_sequence<S, T>(
    spa: &mut S,
    n: u64,
    spa_params: &SPAParams,
    gen_params: &GenerationParams,
    seed_data: Option<&T>,
    output_sequence: &mut T,
) -> Result<f64>
where
    S: GenerationSPA,
    T: Sequence,
{
    let mut loss = 0.0;
    if let Some(data) = seed_data {
        for sym in data.iter() {
            loss += spa.input_seed_data_symbol(sym, spa_params)?;
        }
    }

    let rng_samples = Uniform::new(0.0, 1.0)
        .sample_iter(&mut thread_rng())
        .take(n as usize)
        .collect_vec();

    for i in 0..n {
        let (sym, new_loss) =
            spa.generate_one_symbol(rng_samples[i as usize], spa_params, gen_params)?;
        output_sequence.put_sym(sym)?;
        loss += new_loss;
    }
    spa.cleanup_post_generation();

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use crate::{
        generation::{generate_sequence, GenerationParams},
        sequence::{CharacterSequence, Sequence},
        spa::{DirichletSPA, SPAParams, LZ78SPA, SPA},
    };

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let params = SPAParams::new_lz78_dirichlet(input.alphabet_size(), 0.5, false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &params)
            .expect("failed to train spa");

        let mut generation_output = CharacterSequence::new(input.character_map.clone());

        generate_sequence(
            &mut spa,
            100,
            &params,
            &GenerationParams::new(0.0, 10, 5, 1),
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

        let mut generation_output2 = CharacterSequence::new(input.character_map.clone());

        generate_sequence(
            &mut spa,
            100,
            &params,
            &GenerationParams::new(1.0, 1, 5, 1),
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

        let mut generation_output = CharacterSequence::new(input.character_map.clone());

        generate_sequence(
            &mut spa,
            100,
            &params,
            &GenerationParams::new(2.0, 5, 5, 1),
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

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        generate_sequence(
            &mut spa,
            100,
            &params,
            &GenerationParams::new(0.5, 5, 5, 1),
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
