use anyhow::Result;
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

use crate::{sequence::Sequence, util::sample_from_pdf};

use super::{states::SPAState, SPAParams, SPA};

pub trait GenerationSPA: SPA {
    /// Called when "seeding" the text generation
    fn input_seed_data_symbol(
        &self,
        sym: u32,
        params: &SPAParams,
        gen_state: &mut SPAState,
    ) -> Result<f64>;

    /// Generates one symbol and updates the state of the SPA accordingly.
    fn generate_one_symbol(
        &self,
        rng_sample: f64,
        params: &SPAParams,
        gen_params: &GenerationParams,
        gen_state: &mut SPAState,
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

pub fn gen_symbol_from_spa(
    rng_sample: f64,
    gen_params: &GenerationParams,
    spa: &[f64],
) -> Result<(u32, f64)> {
    let orig_spa = spa;
    let mut spa = spa.to_vec();
    let most_likely_next_sym = (0..spa.len() as u32)
        .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
        .unwrap();

    // if temperature is 0.0, we just compute the argmax of the SPA. If
    // temperature is 1.0, the symbols are generated directly from the
    // SPA. In either case, we do not need the following computation.
    if gen_params.temperature != 0.0 && gen_params.temperature != 1.0 {
        spa = spa
            .iter()
            .map(|x| 2.0_f64.powf(x.log2() / gen_params.temperature))
            .collect_vec();
    }

    // top-k sampling
    (0..spa.len())
        .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
        .take(spa.len() - gen_params.top_k as usize)
        .map(|i| {
            spa[i as usize] = 0.0;
        })
        .collect_vec();

    let sum: f64 = spa.iter().sum();
    spa = spa.iter().map(|x| *x / sum).collect_vec();

    let new_sym = if gen_params.temperature == 0.0 {
        most_likely_next_sym
    } else {
        sample_from_pdf(&spa, rng_sample) as u32
    };
    let loss = -orig_spa[new_sym as usize].log2();
    Ok((new_sym, loss))
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
    let mut gen_state = spa_params.get_new_state(true);

    if let Some(data) = seed_data {
        for sym in data.iter() {
            loss += spa.input_seed_data_symbol(sym, spa_params, &mut gen_state)?;
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
        )?;
        output_sequence.put_sym(sym)?;
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
            SPAParams, SPA,
        },
    };

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let params = SPAParams::new_lz78_dirichlet(input.alphabet_size(), 0.5, false);
        let mut state = params.get_new_state(false);
        let mut spa: LZ78SPA<DirichletSPA> =
            LZ78SPA::new(&params).expect("failed to make LZ78 SPA");
        spa.train_on_block(&input, &params, &mut state)
            .expect("failed to train spa");

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

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

        let mut generation_output2 =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

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

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();

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

        let mut generation_output =
            CharacterSequence::new(&SequenceParams::CharMap(input.character_map.clone())).unwrap();
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
