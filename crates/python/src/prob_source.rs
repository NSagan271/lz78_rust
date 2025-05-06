use lz78::{
    prob_source::{DiracDirichletMixtureTree, DiscreteBinaryThetaSPATree, LZ78Source},
    spa::{
        config::{
            DiracDirichletConfig, DirichletConfigBuilder, DiscreteThetaConfig, LZ78ConfigBuilder,
            SPAConfig,
        },
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

    pub fn get_log_loss(&self) -> f32 {
        self.source.total_log_loss
    }

    pub fn get_n(&self) -> u64 {
        self.source.n
    }

    pub fn get_scaled_log_loss(&self) -> f32 {
        self.source.total_log_loss / self.source.n as f32
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
    #[pyo3(signature = (gamma, dirichlet_weight, dirac_loc, seed = 271))]
    pub fn new(gamma: f64, dirichlet_weight: f64, dirac_loc: f32, seed: u64) -> PyResult<Self> {
        let config = LZ78ConfigBuilder::new(DiracDirichletConfig::new_enum(
            &[0.5, 0.5],
            &[dirac_loc, 1.0 - dirac_loc],
            gamma,
            dirichlet_weight,
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

    pub fn get_log_loss(&self) -> f32 {
        self.source.total_log_loss
    }

    pub fn get_n(&self) -> u64 {
        self.source.n
    }

    pub fn get_scaled_log_loss(&self) -> f32 {
        self.source.total_log_loss / self.source.n as f32
    }
}

#[pyclass]
pub struct DiscreteThetaLZ78Source {
    source: LZ78Source<DiscreteBinaryThetaSPATree>,
    config: SPAConfig,
    state: SPAState,
    rng: StdRng,
}

#[pymethods]
impl DiscreteThetaLZ78Source {
    #[new]
    #[pyo3(signature = (theta_values, probabilities, seed = 271))]
    pub fn new(theta_values: Vec<f32>, probabilities: Vec<f32>, seed: u64) -> PyResult<Self> {
        let config =
            LZ78ConfigBuilder::new(DiscreteThetaConfig::new_enum(&probabilities, &theta_values))
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

    pub fn get_log_loss(&self) -> f32 {
        self.source.total_log_loss
    }

    pub fn get_n(&self) -> u64 {
        self.source.n
    }

    pub fn get_scaled_log_loss(&self) -> f32 {
        self.source.total_log_loss / self.source.n as f32
    }
}
