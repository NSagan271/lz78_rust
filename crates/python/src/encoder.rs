use bytes::Bytes;
use lz78::{
    block_encoder::BlockEncoder,
    encoder::Encoder,
    sequence::{CharacterSequence, Sequence as Sequence_LZ78, U32Sequence, U8Sequence},
};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};

use crate::{sequence::SequenceType, Sequence};

/// Stores an encoded bitstream, as well as some auxiliary information needed
/// for decoding. `CompressedSequence` objects cannot be instantiated directly,
/// but rather are returned by `LZ78Encoder.encode`.
///
/// The main functionality is:
/// 1. Getting the compression ratio as (encoded size) / (uncompressed len * log A),
///     where A is the size of the alphabet.
/// 2. Getting a byte array representing this object, so that the compressed
///     sequence can be stored to a file
#[derive(Clone)]
#[pyclass]
pub struct CompressedSequence {
    encoded_sequence: lz78::encoder::EncodedSequence,
    /// Used for inferring the right type when decoding
    empty_seq_of_correct_datatype: SequenceType,
}

#[pymethods]
impl CompressedSequence {
    /// Returns the compression ratio:  (encoded size) / (uncompressed len * log A),
    /// where A is the size of the alphabet.
    pub fn compression_ratio(&self) -> PyResult<f32> {
        Ok(self.encoded_sequence.compression_ratio())
    }

    /// Returns a byte array representing the compressed sequence.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = self.encoded_sequence.to_bytes();
        bytes.extend(self.empty_seq_of_correct_datatype.to_bytes());
        PyBytes::new_bound(py, &bytes)
    }
}

/// Takes a byte array produced by `CompressedSequence.to_bytes` and returns
/// the corresponding `CompressedSequence` object
#[pyfunction]
pub fn encoded_sequence_from_bytes<'py>(
    bytes: Py<PyBytes>,
    py: Python<'py>,
) -> PyResult<CompressedSequence> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    let encoded_sequence = lz78::encoder::EncodedSequence::from_bytes(&mut bytes);
    let empty_seq_of_correct_datatype = SequenceType::from_bytes(&mut bytes)?;

    Ok(CompressedSequence {
        encoded_sequence,
        empty_seq_of_correct_datatype,
    })
}

/// Encodes and decodes sequences using LZ78 compression
#[pyclass]
pub struct LZ78Encoder {
    encoder: lz78::encoder::LZ8Encoder,
}

#[pymethods]
impl LZ78Encoder {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            encoder: lz78::encoder::LZ8Encoder::new(),
        })
    }

    /// Encodes a `Sequence` object using LZ78 and returns the resulting
    /// `CompressedSequence`. See "Compression of individual sequences via
    /// variable-rate coding" (Ziv, Lempel 1978) for more details.
    fn encode(&self, input: Sequence) -> PyResult<CompressedSequence> {
        let (encoded_sequence, empty_seq_of_correct_datatype) = match input.sequence {
            SequenceType::U8(x) => (
                self.encoder.encode(&x)?,
                SequenceType::U8(U8Sequence::new(x.alphabet_size())),
            ),
            SequenceType::U32(x) => (
                self.encoder.encode(&x)?,
                SequenceType::U32(U32Sequence::new(x.alphabet_size())),
            ),
            SequenceType::Char(x) => (
                self.encoder.encode(&x)?,
                SequenceType::Char(CharacterSequence::new(x.character_map.clone())),
            ),
        };

        Ok(CompressedSequence {
            encoded_sequence,
            empty_seq_of_correct_datatype,
        })
    }

    /// Decodes a sequence compressed via `LZ78Encoder.encode`
    fn decode(&self, input: CompressedSequence) -> PyResult<Sequence> {
        let output = match &input.empty_seq_of_correct_datatype {
            SequenceType::U8(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::U8(seq),
                }
            }
            SequenceType::U32(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::U32(seq),
                }
            }
            SequenceType::Char(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::Char(seq),
                }
            }
        };
        Ok(output)
    }
}

/// Block LZ78 encoder: you can pass in the input sequence to be compressed
/// in chunks, and the output (`encoder.get_encoded_sequence()`) is as if the
/// full concatenated sequence was passed in to an LZ78 encoder
#[derive(Clone)]
#[pyclass]
pub struct BlockLZ78Encoder {
    encoder: lz78::block_encoder::BlockLZ78Encoder,
    empty_seq_of_correct_datatype: Option<SequenceType>,
    alphabet_size: u32,
}

#[pymethods]
impl BlockLZ78Encoder {
    #[new]
    fn new(alpha_size: u32) -> PyResult<Self> {
        Ok(Self {
            encoder: lz78::block_encoder::BlockLZ78Encoder::new(alpha_size),
            empty_seq_of_correct_datatype: None,
            alphabet_size: alpha_size,
        })
    }

    /// Encodes a block using LZ78, starting at the end of the previous block.
    ///
    /// All blocks passed in must be over the same alphabet. For character
    /// sequences, they must use the same `CharacterMap` (i.e., the same chars
    /// are mapped to the same symbols; they need not use the exact same
    /// `CharacterMap` instance).
    ///
    /// The expected alphabet is defined by the first call to `encode_block`,
    /// and subsequent calls will error if the input sequence has a different
    /// alphabet size or character map.
    fn encode_block(&mut self, input: Sequence) -> PyResult<()> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else if self.alphabet_size != input.alphabet_size()? {
            return Err(PyAssertionError::new_err(format!(
                "Expected alphabet size of {}, got {}",
                self.alphabet_size,
                input.alphabet_size()?
            )));
        }

        match input.sequence {
            SequenceType::U8(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U8(U8Sequence::new(x.alphabet_size())));
            }
            SequenceType::U32(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U32(U32Sequence::new(x.alphabet_size())))
            }
            SequenceType::Char(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype = Some(SequenceType::Char(
                    CharacterSequence::new(x.character_map.clone()),
                ))
            }
        };

        Ok(())
    }

    /// Returns the alphabet size passed in upon instantiation
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size
    }

    /// Returns the `CompressedSequence` object, which is equivalent to the
    /// output of `LZ78Encoder.encode` on the concatenation of all inputs to
    /// `encode_block` thus far.
    ///
    /// Errors if no blocks have been compressed so far.
    fn get_encoded_sequence(&mut self) -> PyResult<CompressedSequence> {
        if self.empty_seq_of_correct_datatype.is_none() {
            return Err(PyAssertionError::new_err("No blocks have been encoded yet"));
        }
        Ok(CompressedSequence {
            encoded_sequence: self.encoder.get_encoded_sequence().to_owned(),
            empty_seq_of_correct_datatype: self.empty_seq_of_correct_datatype.clone().unwrap(),
        })
    }

    /// Performs LZ78 decoding on the compressed sequence that has been
    /// generated thus far.
    ///
    /// Errors if no blocks have been compressed so far.
    fn decode(&self) -> PyResult<Sequence> {
        if self.empty_seq_of_correct_datatype.is_none() {
            return Err(PyAssertionError::new_err("No blocks have been encoded yet"));
        }
        let output = match self.empty_seq_of_correct_datatype.clone().unwrap() {
            SequenceType::U8(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::U8(seq),
                }
            }
            SequenceType::U32(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::U32(seq),
                }
            }
            SequenceType::Char(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::Char(seq),
                }
            }
        };
        Ok(output)
    }
}
