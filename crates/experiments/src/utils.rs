use std::{collections::HashMap, fs::File, io::Write};

use anyhow::{anyhow, bail, Result};
use itertools::Itertools;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, Sequence, SequenceParams, U8Sequence},
    spa::causally_processed::ManualQuantizer,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DatasetPartition {
    Train,
    Test,
    Validation,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Losses {
    avg_losses: Vec<f64>,
    ns: Vec<u64>,
}

impl Losses {
    pub fn new(avg_losses: Vec<f64>, ns: Vec<u64>) -> Self {
        Self { avg_losses, ns }
    }

    pub fn save_pickle(&self, filename: &str) -> Result<()> {
        let serialized_losses = serde_pickle::to_vec(&self, Default::default()).unwrap();

        let mut file = File::create(filename)?;
        file.write_all(&serialized_losses)?;

        Ok(())
    }
}

pub fn default_character_map() -> CharacterMap {
    CharacterMap::from_data(
        &"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\n \t.,\"'’?:;-~"
            .to_string(),
    )
}

pub fn default_char_quantizer() -> Result<ManualQuantizer<CharacterSequence>> {
    let orig_map = default_character_map();
    let quant_map = CharacterMap::from_data(&"abcdefghijklmnopqrstuvwxyz0 .~".to_string());
    let mut mapping: HashMap<char, char> = HashMap::new();
    for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
        mapping.insert(char, char.to_ascii_lowercase());
    }
    for char in "1234567890".chars() {
        mapping.insert(char, '0');
    }
    for char in "\n \t".chars() {
        mapping.insert(char, ' ');
    }
    for char in ".,\"'’?:;-".chars() {
        mapping.insert(char, '.');
    }
    mapping.insert('~', '~');

    let mut orig_to_quant_mapping: HashMap<u32, u32> = HashMap::new();
    for (raw, quant) in mapping.into_iter() {
        orig_to_quant_mapping.insert(
            orig_map
                .encode(raw)
                .ok_or_else(|| anyhow!("invalid char"))?,
            quant_map
                .encode(quant)
                .ok_or_else(|| anyhow!("invalid char"))?,
        );
    }

    Ok(ManualQuantizer::<CharacterSequence>::new(
        SequenceParams::CharMap(orig_map),
        SequenceParams::CharMap(quant_map),
        orig_to_quant_mapping,
    ))
}

pub fn quantize_images(images: Vec<Vec<u8>>, quant_bits: u8) -> Vec<Vec<u8>> {
    let quant_strength = 256.0 / 2f32.powf(quant_bits as f32);
    images
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|x| (x as f32 / quant_strength).floor() as u8)
                .collect_vec()
        })
        .collect_vec()
}

pub fn get_default_nucleotide_map() -> HashMap<char, u32> {
    let mut map = HashMap::new();
    map.insert('A', 0);
    map.insert('C', 1);
    map.insert('G', 2);
    map.insert('T', 3);
    map.insert('U', 3);
    map.insert('N', 4);

    map
}

pub fn nucleotides_to_codon(triplet: &[u32]) -> u32 {
    if triplet.contains(&4) || triplet.len() != 3 {
        return 64; // N
    }
    triplet[0] * 16 + triplet[1] * 4 + triplet[2]
}

/// {ACGTN} -> [0-4]
pub fn nucleotides_to_seq(nuc_string: &str) -> Result<U8Sequence> {
    let mut seq = U8Sequence::new(&SequenceParams::AlphaSize(5))?;
    let map = get_default_nucleotide_map();
    for char in nuc_string.chars() {
        let char = char.to_ascii_uppercase();
        if map.contains_key(&char) {
            seq.put_sym(map[&char])?;
        } else {
            bail!("Unexpected character {char} in DNA or RNA string")
        }
    }

    Ok(seq)
}

pub fn nucleotides_to_codon_seqs(nuc_string: &str) -> Result<Vec<U8Sequence>> {
    let map = get_default_nucleotide_map();

    let mut nuc_string = nuc_string.to_ascii_uppercase().replace('T', "U");

    let mut seqs = Vec::new();

    while nuc_string.len() > 0 {
        let start_codon_location = if let Some(x) = nuc_string.find("AUG") {
            x
        } else {
            break;
        };

        let slice = &nuc_string[start_codon_location..];
        let nucleotides = nucleotides_to_seq(slice)?;
        let mut seq = U8Sequence::new(&SequenceParams::AlphaSize(65))?; // all triplets plus N
        let mut next_start = start_codon_location;

        for triplet in &nucleotides.iter().chunks(3) {
            let triplet = triplet.collect_vec();
            next_start += 3;
            if triplet == "UAA".chars().map(|x| map[&x]).collect_vec()
                || triplet == "UAG".chars().map(|x| map[&x]).collect_vec()
                || triplet == "UGA".chars().map(|x| map[&x]).collect_vec()
            {
                break; // stop codon
            }
            seq.put_sym(nucleotides_to_codon(&triplet))?;
        }
        if seq.len() > 0 {
            seqs.push(seq);
        }
        if next_start >= nuc_string.len() {
            break;
        }
        nuc_string = nuc_string[next_start..].to_string();
    }

    Ok(seqs)
}

pub fn print_proteins(proc_codons: &U8Sequence) {
    let proteins = "FLIMVSPTAYHQNKDECWRG".chars().collect_vec();
    for codon in proc_codons.iter() {
        print!("{}", proteins[codon as usize]);
    }
    println!()
}

pub fn get_codon_processor() -> ManualQuantizer<U8Sequence> {
    let mut map: HashMap<u32, u32> = HashMap::new();
    let nuc_map = get_default_nucleotide_map();

    let amino_acids = vec![
        vec!["UUU", "UUC"],                             // Phe
        vec!["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"], // Leu
        vec!["AUU", "AUC", "AUA"],                      // Ile
        vec!["AUG"],                                    // Met
        vec!["GUU", "GUC", "GUA", "GUG"],               // Val
        vec!["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"], // Ser
        vec!["CCU", "CCC", "CCA", "CCG"],               // Pro
        vec!["ACU", "ACC", "ACA", "ACG"],               // Thr
        vec!["GCU", "GCC", "GCA", "GCG"],               // Ala
        vec!["UAU", "UAC"],                             // Tyr
        vec!["CAU", "CAC"],                             // His
        vec!["CAA", "CAG"],                             // Gln
        vec!["AAU", "AAC"],                             // Asn
        vec!["AAA", "AAG"],                             // Lys
        vec!["GAU", "GAC"],                             // Asp
        vec!["GAA", "GAG"],                             // Glu
        vec!["UGU", "UGC"],                             // Cys
        vec!["UGG"],                                    // Trp
        vec!["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"], // Arg
        vec!["GGU", "GGC", "GGA", "GGG"],               // Gly
    ];
    let n_aa = amino_acids.len() as u32;
    map.insert(64, n_aa as u32); // N

    for (i, codons) in amino_acids.into_iter().enumerate() {
        for codon in codons {
            map.insert(
                nucleotides_to_codon(&codon.chars().map(|c| nuc_map[&c]).collect_vec()),
                i as u32,
            );
        }
    }

    println!("{:?}", map.keys().sorted().collect_vec().len());

    ManualQuantizer::new(
        SequenceParams::AlphaSize(65),
        SequenceParams::AlphaSize(n_aa + 1),
        map,
    )
}
