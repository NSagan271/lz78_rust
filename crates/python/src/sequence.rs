use anyhow::bail;
use bytes::{Buf, BufMut, Bytes};
use itertools::Itertools;
use lz78::{
    sequence::{
        CharacterSequence, Sequence as RustSequence, SequenceConfig, U32Sequence, U8Sequence,
    },
    storage::ToFromBytes,
};
use pyo3::{
    exceptions::PyAssertionError,
    prelude::*,
    types::{PyList, PySlice},
};

#[derive(Clone)]
pub enum SequenceType {
    U8(U8Sequence),
    Char(CharacterSequence),
    U32(U32Sequence),
}

impl SequenceType {
    fn alphabet_size(&self) -> u32 {
        match self {
            SequenceType::U8(u8_sequence) => u8_sequence.alphabet_size(),
            SequenceType::Char(character_sequence) => character_sequence.alphabet_size(),
            SequenceType::U32(u32_sequence) => u32_sequence.alphabet_size(),
        }
    }

    fn len(&self) -> u64 {
        match self {
            SequenceType::U8(u8_sequence) => u8_sequence.len(),
            SequenceType::Char(character_sequence) => character_sequence.len(),
            SequenceType::U32(u32_sequence) => u32_sequence.len(),
        }
    }

    fn get(&self, i: u64) -> PyResult<u32> {
        Ok(match self {
            SequenceType::U8(u8_sequence) => u8_sequence.try_get(i)?,
            SequenceType::Char(character_sequence) => character_sequence.try_get(i)?,
            SequenceType::U32(u32_sequence) => u32_sequence.try_get(i)?,
        })
    }

    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            SequenceType::U8(seq) => seq.iter().collect_vec(),
            SequenceType::Char(seq) => seq.iter().collect_vec(),
            SequenceType::U32(seq) => seq.iter().collect_vec(),
        }
    }

    pub fn type_string(&self) -> String {
        match self {
            SequenceType::U8(_) => format!(
                "Byte (U8) Sequence with alphabet size {}",
                self.alphabet_size()
            ),
            SequenceType::Char(c) => format!(
                "String Sequence with character mapping that starts with {:?}",
                c.character_map.sym_to_char.iter().take(5).collect_vec()
            ),
            SequenceType::U32(_) => format!(
                "Integer (U32) Sequence with alphabet size {}",
                self.alphabet_size()
            ),
        }
    }

    pub fn assert_types_match(&self, other: &SequenceType) -> anyhow::Result<()> {
        match self {
            SequenceType::U8(_) => {
                if let SequenceType::U8(_) = other {
                    if self.alphabet_size() == other.alphabet_size() {
                        return Ok(());
                    }
                }
                bail!(
                    "Expected {}, got {}",
                    self.type_string(),
                    other.type_string()
                )
            }
            SequenceType::Char(c) => {
                if let SequenceType::Char(c2) = other {
                    if c.character_map == c2.character_map {
                        return Ok(());
                    }
                }
                bail!(
                    "Expected {}, got {}",
                    self.type_string(),
                    other.type_string()
                )
            }
            SequenceType::U32(_) => {
                if let SequenceType::U32(_) = other {
                    if self.alphabet_size() == other.alphabet_size() {
                        return Ok(());
                    }
                }
                bail!(
                    "Expected {}, got {}",
                    self.type_string(),
                    other.type_string()
                )
            }
        }
    }

    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        match self {
            SequenceType::U8(u8_sequence) => {
                bytes.put_u8(0);
                bytes.put_u32_le(u8_sequence.alphabet_size());
            }
            SequenceType::Char(character_sequence) => {
                bytes.put_u8(1);
                bytes.extend(character_sequence.character_map.to_bytes()?);
            }
            SequenceType::U32(u32_sequence) => {
                bytes.put_u8(2);
                bytes.put_u32_le(u32_sequence.alphabet_size());
            }
        }

        Ok(bytes)
    }

    pub fn from_bytes(bytes: &mut Bytes) -> anyhow::Result<Self> {
        match bytes.get_u8() {
            0 => {
                let alphabet_size = bytes.get_u32_le();
                Ok(Self::U8(U8Sequence::new(&SequenceConfig::AlphaSize(
                    alphabet_size,
                ))?))
            }
            1 => {
                let charmap = lz78::sequence::CharacterMap::from_bytes(bytes)?;
                Ok(Self::Char(CharacterSequence::new(
                    &SequenceConfig::CharMap(charmap),
                )?))
            }
            2 => {
                let alphabet_size = bytes.get_u32_le();
                Ok(Self::U32(U32Sequence::new(&SequenceConfig::AlphaSize(
                    alphabet_size,
                ))?))
            }
            _ => bail!("error parsing SequenceType"),
        }
    }
}

/// A Sequence is a list of strings or integers that can be encoded by LZ78.
/// Each sequence is associated with an alphabet size, A.
///
/// If the sequence consists of integers, they must be in the range
/// {0, 1, ..., A-1}. If A < 256, the sequence is stored internally as bytes.
/// Otherwise, it is stored as `uint32`.
///
/// If the sequence is a string, a `CharacterMap` object maps each
/// character to a number between 0 and A-1.
///
/// Inputs:
/// - data: either a list of integers or a string.
/// - alphabet_size (if data is list[int]): the size of the alphabet. Must be
///      specified if `data` is a list of integers, and is ignored if `data`
///     is a string.
/// - charmap (if data is string): A `CharacterMap` object; must be specified
///     if `data` is a string, and is ignored if `data` is numerical.
///
#[pyclass]
#[derive(Clone)]
pub struct Sequence {
    pub sequence: SequenceType,
}

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (data, alphabet_size=None, charmap=None))]
    pub fn new<'py>(
        data: Bound<'py, PyAny>,
        alphabet_size: Option<u32>,
        charmap: Option<CharacterMap>,
    ) -> PyResult<Self> {
        // check if this is a string
        let maybe_input_string = data.extract::<String>();
        if maybe_input_string.is_ok() {
            if charmap.is_none() {
                return Err(PyAssertionError::new_err(
                    "MUST specify a CharacterMap to instantiate a string sequence.\nUsage: seq = Sequence(\"my string\", charmap=CharacterMap(\"abcd...\"))"
                ));
            }
            let cmap = charmap.unwrap();
            return Ok(Self {
                sequence: SequenceType::Char(CharacterSequence::from_data(
                    maybe_input_string.unwrap(),
                    cmap.map,
                )?),
            });
        }

        if alphabet_size.is_none() {
            return Err(PyAssertionError::new_err(
                "MUST specify an alphabet size to instantiate a numerical sequence.\nUsage: seq = Sequence([1, 2, 3, 4, 1, 2, 3, 4, 0], alphabet_size=5)"
            ));
        }

        let alphabet_size = alphabet_size.unwrap();
        if alphabet_size <= 256 {
            return Ok(Self {
                sequence: SequenceType::U8(U8Sequence::from_data(
                    data.extract::<Vec<u8>>()?,
                    alphabet_size,
                )?),
            });
        } else {
            return Ok(Self {
                sequence: SequenceType::U32(U32Sequence::from_data(
                    data.extract::<Vec<u32>>()?,
                    alphabet_size,
                )?),
            });
        }
    }

    /// Extend the sequence with new data, which must have the same alphabet
    /// as the current sequence. If this sequence is represented by a string,
    /// then `data` will be encoded using the same character map as the
    /// current sequence
    pub fn extend<'py>(&mut self, data: Bound<'py, PyAny>) -> PyResult<()> {
        match &mut self.sequence {
            SequenceType::U8(u8_sequence) => {
                let new_seq = data.extract::<Vec<u8>>()?;
                u8_sequence.extend(&new_seq)?;
            }
            SequenceType::Char(character_sequence) => {
                let new_seq = data.extract::<String>()?;
                character_sequence.extend(&new_seq)?;
            }
            SequenceType::U32(u32_sequence) => {
                let new_seq = data.extract::<Vec<u32>>()?;
                u32_sequence.extend(&new_seq)?;
            }
        }

        Ok(())
    }

    /// Returns the size of the sequence's alphabet
    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.sequence.alphabet_size())
    }

    /// If this sequence is represented by a string, returns the underlying
    /// object that maps characters to integers. Otherwise, this will error.
    pub fn get_character_map(&self) -> PyResult<CharacterMap> {
        match &self.sequence {
            SequenceType::Char(character_sequence) => Ok(CharacterMap {
                map: character_sequence.character_map.clone(),
            }),
            _ => {
                return Err(PyAssertionError::new_err(
                    "Tried to get a character map from an integer sequence",
                ))
            }
        }
    }

    /// Fetches the raw data (as a list of integers, or a string) underlying
    /// this sequence
    pub fn get_data(&self, py: Python) -> PyObject {
        match &self.sequence {
            SequenceType::U8(u8_sequence) => u8_sequence.data.to_object(py),
            SequenceType::Char(character_sequence) => character_sequence.data.to_object(py),
            SequenceType::U32(u32_sequence) => u32_sequence.data.to_object(py),
        }
    }

    fn __len__(&self) -> usize {
        self.sequence.len() as usize
    }

    fn __getitem__<'py>(&self, py: Python<'py>, i: Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(i) = i.extract::<isize>() {
            let i = if i < 0 {
                (self.__len__() as isize + i) as u64
            } else {
                i as u64
            };
            return Ok(self.sequence.get(i)?.to_object(py));
        } else if i.is_instance_of::<PySlice>() {
            // handle indexing by a slice
            let res = PyList::empty_bound(py);

            // start defaults to 0
            let mut start = if i.getattr("start")?.is_none() {
                0
            } else {
                i.getattr("start")?.extract::<isize>()?
            };
            if start < 0 {
                start = self.__len__() as isize + start;
            }

            // stop defaults to the end of the sequence
            let mut stop = if i.getattr("stop")?.is_none() {
                self.__len__() as isize
            } else {
                i.getattr("stop")?.extract::<isize>()?
            };
            if stop < 0 {
                stop = self.__len__() as isize + stop;
            }

            // step defaults to 1
            let step = if i.getattr("step")?.is_none() {
                1
            } else {
                i.getattr("step")?.extract::<isize>()?
            };

            for idx in (start..stop).step_by(step.abs() as usize) {
                res.append(self.sequence.get(idx as u64)?)?;
            }
            if step < 0 {
                res.reverse()?;
            }

            return Ok(res.unbind().into_any());
        }
        Err(PyAssertionError::new_err(
            "Expected a slice or integer index",
        ))
    }
}

/// Maps characters in a string to uint32 values in a contiguous range, so that
/// a string can be used as an individual sequence. Has the capability to
/// **encode** a string into the corresponding integer representation, and
/// **decode** a list of integers into a string.
///
/// Inputs:
/// - data: a string consisting of all of the characters that will appear in
///     the character map. For instance, a common use case is:
///     ```
///     charmap = CharacterMap("abcdefghijklmnopqrstuvwxyz")
///     ```
#[pyclass]
#[derive(Clone)]
pub struct CharacterMap {
    map: lz78::sequence::CharacterMap,
}

#[pymethods]
impl CharacterMap {
    #[new]
    pub fn new(data: String) -> PyResult<Self> {
        Ok(Self {
            map: lz78::sequence::CharacterMap::from_data(&data),
        })
    }

    /// Given a string, returns its encoding as a list of integers
    pub fn encode(&self, s: String) -> PyResult<Vec<u32>> {
        Ok(self.map.try_encode_all(&s)?)
    }

    /// Given a list of integers between 0 and self.alphabet_size() - 1, return
    /// the corresponding string representation
    pub fn decode(&self, syms: Vec<u32>) -> PyResult<String> {
        Ok(self.map.try_decode_all(syms)?)
    }

    /// Given a string, filter out all characters that aren't part of the
    /// mapping and return the resulting string
    pub fn filter_string(&self, data: String) -> PyResult<String> {
        Ok(self.map.filter_string(&data))
    }

    /// Add a new character to the mapping, if it is not already there.
    pub fn add(&mut self, c: char) {
        self.map.add(c);
    }

    /// Given a string, filter out all characters that aren't part of the
    /// mapping and replace them with the specified char (which must be
    /// a part of the character map)
    pub fn filter_string_and_replace(&self, data: String, c: char) -> PyResult<String> {
        Ok(self.map.filter_string_and_replace(&data, c))
    }

    /// Returns the number of characters that can be represented by this map
    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.map.alphabet_size)
    }
}
