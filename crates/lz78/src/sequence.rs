use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};

pub trait Sequence {
    fn alphabet_size(&self) -> u32;

    fn len(&self) -> u64;

    fn get(&self, i: u64) -> Result<u32>;

    fn put_sym(&mut self, sym: u32);
}

pub struct BinarySequence {
    pub data: Vec<u8>,
}

impl Sequence for BinarySequence {
    fn alphabet_size(&self) -> u32 {
        2
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u8);
    }
}

impl BinarySequence {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn from_data(data: Vec<u8>) -> Result<Self> {
        if data.iter().any(|x| *x > 1) {
            bail!("input to BinarySequence.extend has symbols that are not 0 or 1");
        }
        Ok(Self { data })
    }

    pub fn extend(&mut self, data: &[u8]) -> Result<()> {
        if data.iter().any(|x| *x > 1) {
            bail!("input to BinarySequence.extend has symbols that are not 0 or 1");
        }
        self.data.extend(data);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CharacterMap {
    char_to_sym: HashMap<String, u32>,
    sym_to_char: Vec<String>,
    pub alphabet_size: u32,
}

impl CharacterMap {
    pub fn from_data(data: &String) -> Self {
        let mut char_to_sym: HashMap<String, u32> = HashMap::new();
        let mut charset: HashSet<String> = HashSet::new();
        let mut alphabet_size = 0;
        let mut sym_to_char: Vec<String> = Vec::new();

        for char in data.chars() {
            let key = char.to_string();
            if !charset.contains(&key) {
                char_to_sym.insert(key.clone(), alphabet_size);
                charset.insert(key.clone());
                sym_to_char.push(key);
                alphabet_size += 1;
            }
        }

        Self {
            alphabet_size,
            char_to_sym,
            sym_to_char,
        }
    }

    pub fn encode(&self, char: &String) -> Option<u32> {
        if self.char_to_sym.contains_key(char) {
            Some(self.char_to_sym[char])
        } else {
            None
        }
    }

    pub fn encode_all(&self, data: &String) -> Result<Vec<u32>> {
        let mut res = Vec::new();
        for char in data.chars() {
            res.push(
                self.encode(&char.to_string())
                    .ok_or(anyhow!("invalid char"))?,
            );
        }
        Ok(res)
    }

    pub fn filter_string(&self, data: &String) -> String {
        let mut filt = String::with_capacity(data.len());
        for char in data.chars() {
            let key = char.to_string();
            filt.push_str(if self.char_to_sym.contains_key(&key) {
                &key
            } else {
                ""
            });
        }
        filt
    }

    pub fn decode(&self, sym: u32) -> Option<String> {
        if sym < self.alphabet_size {
            Some(self.sym_to_char[sym as usize].clone())
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct CharacterSequence {
    pub data: String,
    encoded: Vec<u32>,
    pub character_map: CharacterMap,
}

impl Sequence for CharacterSequence {
    fn alphabet_size(&self) -> u32 {
        self.character_map.alphabet_size
    }

    fn len(&self) -> u64 {
        self.encoded.len() as u64
    }

    fn get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.encoded[i as usize])
    }

    fn put_sym(&mut self, sym: u32) {
        self.data
            .push_str(&self.character_map.decode(sym).unwrap_or("".to_string()));
        self.encoded.push(sym);
    }
}

impl CharacterSequence {
    pub fn new(character_map: CharacterMap) -> Self {
        Self {
            data: String::new(),
            character_map,
            encoded: Vec::new(),
        }
    }

    pub fn from_data(data: String, character_map: CharacterMap) -> Result<Self> {
        for char in data.chars() {
            if character_map.encode(&char.to_string()).is_none() {
                bail!("Invalid symbol in input: {}", &char);
            }
        }
        Ok(Self {
            encoded: character_map.encode_all(&data)?,
            data,
            character_map,
        })
    }

    pub fn from_data_inferred_character_map(data: String) -> Self {
        let character_map = CharacterMap::from_data(&data);
        Self {
            encoded: character_map.encode_all(&data).unwrap(),
            character_map,
            data,
        }
    }

    pub fn from_data_filtered(data: String, character_map: CharacterMap) -> Self {
        let data = character_map.filter_string(&data);
        Self {
            encoded: character_map.encode_all(&data).unwrap(),
            data,
            character_map,
        }
    }
}

#[derive(Clone)]
pub struct U8Sequence {
    pub data: Vec<u8>,
    alphabet_size: u32,
}

impl Sequence for U8Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size as u32
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u8);
    }
}

impl U8Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u8>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x as u32 > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u8>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) as u32 + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u8]) -> Result<()> {
        if data.iter().any(|x| *x as u32 > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

pub struct U16Sequence {
    pub data: Vec<u16>,
    alphabet_size: u32,
}

impl Sequence for U16Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u16);
    }
}

impl U16Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u16>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x as u32 > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u16>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) as u32 + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u16]) -> Result<()> {
        if data.iter().any(|x| *x as u32 > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

#[derive(Clone)]
pub struct U32Sequence {
    pub data: Vec<u32>,
    alphabet_size: u32,
}

impl Sequence for U32Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size as u32
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u32);
    }
}

impl U32Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u32>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u32>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u32]) -> Result<()> {
        if data.iter().any(|x| *x > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::CharacterMap;

    #[test]
    fn test_charmap() {
        let charmap = CharacterMap::from_data(&"abcdefghijklmnopqrstuvwxyz ".to_string());
        assert_eq!(charmap.alphabet_size, 27);
        assert_eq!(charmap.encode(&"j".to_string()), Some(9));
        assert_eq!(charmap.encode(&"z".to_string()), Some(25));
        assert!(charmap.encode(&"1".to_string()).is_none());

        assert_eq!(
            charmap.filter_string(&"hello world 123".to_string()),
            "hello world ".to_string()
        );

        assert_eq!(charmap.decode(1), Some("b".to_string()));
        assert_eq!(charmap.decode(26), Some(" ".to_string()));
        assert_eq!(charmap.decode(24), Some("y".to_string()));
        assert!(charmap.decode(27).is_none());
    }
}