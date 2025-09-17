use hashbrown::HashMap;
use lz78::kmer::get_background_probs;
use lz78::kmer::KmerMultinomial as RustKmerMultinomial;
use pyo3::pyclass;
use pyo3::{exceptions::PyAssertionError, prelude::*};

#[pyclass]
#[derive(Clone)]
pub struct BackgroundPriors {
    pub canonical: bool,
    pub priors: HashMap<u16, HashMap<u64, f64>>, // maps k to prior for kmers
}

#[pymethods]
impl BackgroundPriors {
    #[new]
    pub fn new(canonical: bool) -> Self {
        BackgroundPriors {
            canonical,
            priors: HashMap::new(),
        }
    }

    pub fn add_from_file(&mut self, k: u16, filename: &str) -> PyResult<()> {
        self.priors.insert(
            k,
            get_background_probs(filename, k as usize, true)
                .ok_or_else(|| PyAssertionError::new_err("No k-mers found in file"))?,
        );

        Ok(())
    }

    pub fn has_k(&self, k: u16) -> bool {
        self.priors.contains_key(&k)
    }
}

impl BackgroundPriors {
    pub fn get(&self, k: u16) -> Option<&HashMap<u64, f64>> {
        if let Some(prior) = self.priors.get(&k) {
            Some(prior)
        } else {
            None
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Sequences {
    pub sequences: Vec<String>,
}

#[pymethods]
impl Sequences {
    #[new]
    pub fn new(sequences: Vec<String>) -> Self {
        Sequences { sequences }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct KmerMultinomial {
    pub inner: RustKmerMultinomial,
}

#[pymethods]
impl KmerMultinomial {
    #[new]
    #[pyo3(signature = (k, alpha, feature_mode="count", canonical=true, background_priors=None))]
    pub fn new(
        k: usize,
        alpha: f64,
        feature_mode: &str,
        canonical: bool,
        background_priors: Option<&BackgroundPriors>,
    ) -> PyResult<Self> {
        let fm = match feature_mode {
            "count" => lz78::kmer::FeatureMode::Count,
            "binary" => lz78::kmer::FeatureMode::Binary,
            _ => {
                return Err(PyAssertionError::new_err(
                    "feature_mode must be 'count' or 'binary'",
                ))
            }
        };
        let bg_priors = if let Some(bp) = background_priors {
            bp.get(k as u16).cloned()
        } else {
            None
        };
        Ok(KmerMultinomial {
            inner: RustKmerMultinomial::new(k, alpha, fm, canonical, bg_priors),
        })
    }
    pub fn fit_from_files(&mut self, path_per_class: Vec<String>) -> PyResult<()> {
        self.inner
            .fit_from_files(path_per_class.iter().map(|s| s.as_str()).collect());
        Ok(())
    }

    pub fn predict_one(&self, seq: &str) -> u64 {
        self.inner.predict_one(seq)
    }

    pub fn predict_parallel(&self, seqs: &Sequences, num_threads: usize) -> Vec<u64> {
        self.inner.predict_parallel(
            seqs.sequences.iter().map(|s| s.as_str()).collect(),
            num_threads,
        )
    }
}
