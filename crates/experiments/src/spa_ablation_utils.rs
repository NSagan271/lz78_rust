use anyhow::{bail, Result};
use bytes::{Buf, BufMut};
use lz78::{
    sequence::{CharacterSequence, Sequence, SequenceParams, U8Sequence},
    spa::{
        basic_spas::DirichletSPA,
        causally_processed::{
            generate_sequence_causally_processed, CausalProcessor, CausallyProcessedLZ78SPA,
            CausallyProcessedLZ78SPAParams, IntegerScalarQuantizer, ManualQuantizer,
        },
        ctw::CTW,
        generation::{generate_sequence, GenerationParams},
        lz_transform::LZ78SPA,
        SPAParams,
    },
    storage::ToFromBytes,
};

pub enum SPATypes {
    Dirichlet(DirichletSPA, SPAParams, SequenceParams),
    LZ78Dirichlet(LZ78SPA<DirichletSPA>, SPAParams, SequenceParams),
    LZ78CTW(LZ78SPA<CTW>, SPAParams, SequenceParams),
    ScalarQuantizedLZ78(
        CausallyProcessedLZ78SPA<DirichletSPA>,
        IntegerScalarQuantizer<U8Sequence>,
        CausallyProcessedLZ78SPAParams,
    ),
    CharQuantizedLZ78(
        CausallyProcessedLZ78SPA<DirichletSPA>,
        ManualQuantizer<CharacterSequence>,
        CausallyProcessedLZ78SPAParams,
    ),
}

impl SPATypes {
    pub fn get_seed_data_and_empty_string_seq(
        seq_params: &mut SequenceParams,
        seed_data: &Option<String>,
    ) -> Result<(CharacterSequence, CharacterSequence)> {
        let generate_output = CharacterSequence::new(seq_params)?;
        let seed_data = if let SequenceParams::CharMap(charmap) = seq_params {
            CharacterSequence::from_data_filtered(
                seed_data.clone().unwrap_or("".to_string()),
                charmap.clone(),
            )
        } else {
            bail!("unexpected sequence parameters for string generation")
        };
        Ok((seed_data, generate_output))
    }

    pub fn generate_string(
        &mut self,
        n: u64,
        gen_params: &GenerationParams,
        seed_data: &Option<String>,
    ) -> Result<()> {
        match self {
            SPATypes::Dirichlet(spa, spa_params, seq_params) => {
                let (seed_data, mut gen_output) =
                    Self::get_seed_data_and_empty_string_seq(seq_params, seed_data)?;
                generate_sequence(
                    spa,
                    n,
                    &spa_params,
                    &gen_params,
                    Some(&seed_data),
                    &mut gen_output,
                )?;
                println!("{}{}", seed_data.data, gen_output.data);
            }
            SPATypes::LZ78Dirichlet(spa, spa_params, seq_params) => {
                let (seed_data, mut gen_output) =
                    Self::get_seed_data_and_empty_string_seq(seq_params, seed_data)?;

                generate_sequence(
                    spa,
                    n,
                    &spa_params,
                    &gen_params,
                    Some(&seed_data),
                    &mut gen_output,
                )?;
                println!("{}{}", seed_data.data, gen_output.data);
            }
            SPATypes::LZ78CTW(spa, spa_params, seq_params) => {
                let (seed_data, mut gen_output) =
                    Self::get_seed_data_and_empty_string_seq(seq_params, seed_data)?;

                generate_sequence(
                    spa,
                    n,
                    &spa_params,
                    &gen_params,
                    Some(&seed_data),
                    &mut gen_output,
                )?;
                println!("{}{}", seed_data.data, gen_output.data);
            }
            SPATypes::ScalarQuantizedLZ78(_, _, _) => {
                bail!("string generation not supported for ScalarQuantizedLZ78")
            }
            SPATypes::CharQuantizedLZ78(spa, quant, spa_params) => {
                let (seed_data, gen_output) =
                    Self::get_seed_data_and_empty_string_seq(&mut quant.orig_params, seed_data)?;
                let seed_data = quant.get_causally_processed_seq(seed_data)?;
                let mut gen_output = quant.get_causally_processed_seq(gen_output)?;

                generate_sequence_causally_processed(
                    spa,
                    n,
                    spa_params,
                    gen_params,
                    quant,
                    Some(&seed_data),
                    &mut gen_output,
                )?;
                println!("{}{}", seed_data.original.data, gen_output.original.data);
            }
        }
        Ok(())
    }
}

impl ToFromBytes for SPATypes {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SPATypes::Dirichlet(spa, spa_params, seq_params) => {
                bytes.put_u8(0);
                bytes.extend(spa.to_bytes()?);
                bytes.extend(spa_params.to_bytes()?);
                bytes.extend(seq_params.to_bytes()?);
            }
            SPATypes::LZ78Dirichlet(spa, spa_params, seq_params) => {
                bytes.put_u8(1);
                bytes.extend(spa.to_bytes()?);
                bytes.extend(spa_params.to_bytes()?);
                bytes.extend(seq_params.to_bytes()?);
            }
            SPATypes::ScalarQuantizedLZ78(
                causally_processed_lz78_spa,
                integer_scalar_quantizer,
                causally_processed_lz78_spaparams,
            ) => {
                bytes.put_u8(2);
                bytes.extend(causally_processed_lz78_spa.to_bytes()?);
                bytes.extend(integer_scalar_quantizer.to_bytes()?);
                bytes.extend(causally_processed_lz78_spaparams.to_bytes()?);
            }
            SPATypes::CharQuantizedLZ78(
                causally_processed_lz78_spa,
                manual_quantizer,
                causally_processed_lz78_spaparams,
            ) => {
                bytes.put_u8(3);
                bytes.extend(causally_processed_lz78_spa.to_bytes()?);
                bytes.extend(manual_quantizer.to_bytes()?);
                bytes.extend(causally_processed_lz78_spaparams.to_bytes()?);
            }
            SPATypes::LZ78CTW(spa, spa_params, seq_params) => {
                bytes.put_u8(4);
                bytes.extend(spa.to_bytes()?);
                bytes.extend(spa_params.to_bytes()?);
                bytes.extend(seq_params.to_bytes()?);
            }
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match bytes.get_u8() {
            0 => {
                let spa = DirichletSPA::from_bytes(bytes)?;
                let params = SPAParams::from_bytes(bytes)?;
                let seq_params = SequenceParams::from_bytes(bytes)?;
                Self::Dirichlet(spa, params, seq_params)
            }
            1 => {
                let spa = LZ78SPA::<DirichletSPA>::from_bytes(bytes)?;
                let params = SPAParams::from_bytes(bytes)?;
                let seq_params = SequenceParams::from_bytes(bytes)?;
                Self::LZ78Dirichlet(spa, params, seq_params)
            }
            2 => {
                let spa = CausallyProcessedLZ78SPA::<DirichletSPA>::from_bytes(bytes)?;
                let quant = IntegerScalarQuantizer::<U8Sequence>::from_bytes(bytes)?;
                let params = CausallyProcessedLZ78SPAParams::from_bytes(bytes)?;
                Self::ScalarQuantizedLZ78(spa, quant, params)
            }
            3 => {
                let spa = CausallyProcessedLZ78SPA::<DirichletSPA>::from_bytes(bytes)?;
                let quant = ManualQuantizer::<CharacterSequence>::from_bytes(bytes)?;
                let params = CausallyProcessedLZ78SPAParams::from_bytes(bytes)?;
                Self::CharQuantizedLZ78(spa, quant, params)
            }
            4 => {
                let spa = LZ78SPA::<CTW>::from_bytes(bytes)?;
                let params = SPAParams::from_bytes(bytes)?;
                let seq_params = SequenceParams::from_bytes(bytes)?;
                Self::LZ78CTW(spa, params, seq_params)
            }
            _ => bail!("Unexpected SPAType"),
        })
    }
}
