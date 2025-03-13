use lz78::{
    prob_source::{DiracDirichletMixtureTree, LZ78Source},
    spa::{
        config::{DiracDirichletConfig, DirichletConfigBuilder, LZ78ConfigBuilder, SPAConfig},
        dirichlet::DirichletSPATree,
        states::SPAState,
    },
};
use pyo3::{pyclass, pymethods, PyResult};
use rand::{rngs::StdRng, SeedableRng};

#[pyclass]
pub struct DirichletLZ78Source {
    source: LZ78Source<DirichletSPATree>,
    config: SPAConfig,
    state: SPAState,
    rng: StdRng,
}

#[pymethods]
impl DirichletLZ78Source {
    #[new]
    #[pyo3(signature = (alphabet_size, gamma, seed = 271))]
    pub fn new(alphabet_size: u32, gamma: f64, seed: u64) -> PyResult<Self> {
        let config = LZ78ConfigBuilder::new(
            DirichletConfigBuilder::new(alphabet_size)
                .gamma(gamma)
                .build_enum(),
        )
        .build_enum();
        let mut rng = StdRng::seed_from_u64(seed);
        let source = LZ78Source::new(&config, &mut rng)?;
        let state = SPAState::get_new_state(&config);

        Ok(Self {
            source,
            config,
            state,
            rng,
        })
    }

    pub fn generate_symbols(&mut self, n: u64) -> PyResult<Vec<u32>> {
        let syms =
            self.source
                .generate_symbols(n, &mut self.rng, &mut self.config, &mut self.state)?;
        Ok(syms.data)
    }

    pub fn get_log_loss(&self) -> f64 {
        self.source.total_log_loss
    }

    pub fn get_n(&self) -> u64 {
        self.source.n
    }

    pub fn get_scaled_log_loss(&self) -> f64 {
        self.source.total_log_loss / self.source.n as f64
    }
}

#[pyclass]
pub struct DiracDirichletLZ78Source {
    source: LZ78Source<DiracDirichletMixtureTree>,
    config: SPAConfig,
    state: SPAState,
    rng: StdRng,
}

#[pymethods]
impl DiracDirichletLZ78Source {
    #[new]
    #[pyo3(signature = (gamma, delta, dirac_loc, seed = 271))]
    pub fn new(gamma: f64, delta: f64, dirac_loc: f64, seed: u64) -> PyResult<Self> {
        let config = LZ78ConfigBuilder::new(DiracDirichletConfig::new_enum(
            &[0.5, 0.5],
            &[dirac_loc, 1.0 - dirac_loc],
            gamma,
            delta,
        ))
        .build_enum();
        let mut rng = StdRng::seed_from_u64(seed);
        let source = LZ78Source::new(&config, &mut rng)?;
        let state = SPAState::get_new_state(&config);

        Ok(Self {
            source,
            config,
            state,
            rng,
        })
    }

    pub fn generate_symbols(&mut self, n: u64) -> PyResult<Vec<u32>> {
        let syms =
            self.source
                .generate_symbols(n, &mut self.rng, &mut self.config, &mut self.state)?;
        Ok(syms.data)
    }

    pub fn get_log_loss(&self) -> f64 {
        self.source.total_log_loss
    }

    pub fn get_n(&self) -> u64 {
        self.source.n
    }

    pub fn get_scaled_log_loss(&self) -> f64 {
        self.source.total_log_loss / self.source.n as f64
    }
}
