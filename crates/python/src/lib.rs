pub mod encoder;
pub mod sequence;
pub mod spa;
use encoder::*;
use pyo3::prelude::*;
use sequence::*;
use spa::{spa_from_bytes, spa_from_file, LZ78SPA};

#[pymodule]
fn lz78(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CharacterMap>()?;
    m.add_class::<Sequence>()?;
    m.add_class::<CompressedSequence>()?;
    m.add_class::<LZ78Encoder>()?;
    m.add_class::<BlockLZ78Encoder>()?;
    m.add_class::<LZ78SPA>()?;
    m.add_function(wrap_pyfunction!(spa_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(spa_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(encoded_sequence_from_bytes, m)?)?;
    Ok(())
}
