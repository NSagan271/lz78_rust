use std::{collections::HashMap, sync::Arc};

use anyhow::{bail, Result};
use bytes::{Buf, BufMut, Bytes};
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::storage::ToFromBytes;

use super::SPAConfig;

pub const LZ_ROOT_IDX: u64 = 0;

#[derive(Clone)]
pub enum SPAState {
    LZ78(LZ78State),
    None,
}

impl SPAState {
    pub fn get_new_state(config: &SPAConfig) -> Self {
        match config {
            SPAConfig::LZ78(config) => {
                let root_state = Self::get_new_state(&config.inner_config);
                let child_states = if let Self::None = root_state {
                    None
                } else {
                    let mut map = HashMap::new();
                    map.insert(LZ_ROOT_IDX, root_state);
                    Some(map)
                };

                let state = LZ78State {
                    node: LZ_ROOT_IDX,
                    depth: 0,
                    child_states,
                    patches: Patches::new(),
                    tree_debug: TreeDebug::new(),
                    ensemble: Vec::new(),
                    next_ensemble_offset: 0,
                };
                Self::LZ78(state)
            }
            _ => Self::None,
        }
    }

    pub fn reset(&mut self) {
        match self {
            SPAState::LZ78(state) => state.reset(),
            _ => {}
        }
    }

    pub fn try_get_lz78(&mut self) -> Result<&mut LZ78State> {
        match self {
            Self::LZ78(state) => Ok(state),
            _ => bail!("Invalid state for LZ78 SPA"),
        }
    }
}

impl ToFromBytes for SPAState {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        match self {
            SPAState::LZ78(state) => {
                bytes.put_u8(0);
                bytes.extend(state.to_bytes()?);
            }
            SPAState::None => {
                bytes.put_u8(2);
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match bytes.get_u8() {
            0 => SPAState::LZ78(LZ78State::from_bytes(bytes)?),
            2 => SPAState::None,
            _ => bail!("Invalid SPAState type"),
        })
    }
}

#[derive(Clone)]
pub struct Patches {
    pub patch_information: Vec<(u64, u64)>, // tuple of (start, end), where end is exclusive
    pub store_patches: bool,
    pub internal_counter: u64,
}

impl Patches {
    fn new() -> Self {
        Self {
            patch_information: Vec::new(),
            store_patches: false,
            internal_counter: 0,
        }
    }
    fn reset(&mut self) {
        self.patch_information.clear();
        self.internal_counter = 0;
    }
}

impl ToFromBytes for Patches {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.patch_information.len() as u64);
        for (a, b) in self.patch_information.iter() {
            bytes.put_u64_le(*a);
            bytes.put_u64_le(*b);
        }

        bytes.put_u8(self.store_patches as u8);

        bytes.put_u64_le(self.internal_counter);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n_patch = bytes.get_u64_le() as usize;
        let mut patch_information = Vec::with_capacity(n_patch);
        for _ in 0..n_patch {
            patch_information.push((bytes.get_u64_le(), bytes.get_u64_le()));
        }

        let store_patches = bytes.get_u8() > 0;

        let internal_counter = bytes.get_u64_le();

        Ok(Self {
            patch_information,
            store_patches,
            internal_counter,
        })
    }
}

#[derive(Clone)]
pub struct TreeDebug {
    pub leaf_depths: Vec<u32>,
    pub store_leaf_depths: bool,
}

impl TreeDebug {
    pub fn new() -> Self {
        Self {
            leaf_depths: Vec::new(),
            store_leaf_depths: false,
        }
    }

    pub fn reset(&mut self) {
        self.leaf_depths.clear();
        self.store_leaf_depths = false;
    }
}

impl ToFromBytes for TreeDebug {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.leaf_depths.to_bytes()?;
        bytes.put_u8(self.store_leaf_depths as u8);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let leaf_depths = Vec::<u32>::from_bytes(bytes)?;
        let store_leaf_depths = bytes.get_u8() > 0;
        Ok(Self {
            leaf_depths,
            store_leaf_depths,
        })
    }
}

#[derive(Clone, Copy)]
pub struct NodeAndDepth {
    pub node: u64,
    pub depth: u32,
}

impl NodeAndDepth {
    pub fn go_to_root(&mut self) {
        self.node = LZ_ROOT_IDX;
        self.depth = 0;
    }

    pub fn root() -> Self {
        Self {
            node: LZ_ROOT_IDX,
            depth: 0,
        }
    }
}

impl ToFromBytes for NodeAndDepth {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.node);
        bytes.put_u32_le(self.depth);
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let node = bytes.get_u64_le();
        let depth = bytes.get_u32_le();
        Ok(Self { node, depth })
    }
}

impl ToFromBytes for Vec<NodeAndDepth> {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.len() as u64);
        for val in self.iter() {
            bytes.extend(val.to_bytes()?);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le() as usize;
        let mut res = Vec::with_capacity(n);
        for _ in 0..n {
            res.push(NodeAndDepth::from_bytes(bytes)?);
        }
        Ok(res)
    }
}

#[derive(Clone)]
pub struct LZ78State {
    pub node: u64,
    pub depth: u32,
    pub child_states: Option<HashMap<u64, SPAState>>,
    pub patches: Patches,
    pub tree_debug: TreeDebug,
    pub ensemble: Vec<NodeAndDepth>,
    pub next_ensemble_offset: u32,
}

impl LZ78State {
    pub fn new(node: u64, depth: u32, child_states: Option<HashMap<u64, SPAState>>) -> Self {
        Self {
            node,
            depth,
            child_states,
            patches: Patches::new(),
            tree_debug: TreeDebug::new(),
            ensemble: Vec::new(),
            next_ensemble_offset: 0,
        }
    }
    pub fn get_child_state(&mut self, child_config: &SPAConfig) -> Option<&mut SPAState> {
        if let Some(children) = &mut self.child_states {
            if !children.contains_key(&self.node) {
                children.insert(self.node, SPAState::get_new_state(child_config));
            }
            children.get_mut(&self.node)
        } else {
            None
        }
    }

    pub fn go_to_root(&mut self) {
        self.node = LZ_ROOT_IDX;
        self.depth = 0;
        self.ensemble.clear();
        self.next_ensemble_offset = 0;
    }

    pub fn reset(&mut self) {
        self.go_to_root();
        if let Some(children) = &mut self.child_states {
            children.clear();
        }
        self.patches.reset();
        self.tree_debug.reset();
    }

    pub fn toggle_store_patches(&mut self, store_patches: bool) {
        self.patches.store_patches = store_patches;
    }

    pub fn clear_patches(&mut self) {
        self.patches.patch_information.clear();
    }
}

impl ToFromBytes for LZ78State {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.node);
        bytes.put_u32_le(self.depth);
        bytes.put_u8(self.child_states.is_some() as u8);
        if let Some(child_states) = &self.child_states {
            bytes.put_u64_le(child_states.len() as u64);
            for (&node, state) in child_states {
                bytes.put_u64_le(node);
                bytes.extend(state.to_bytes()?);
            }
        }

        bytes.extend(self.patches.to_bytes()?);
        bytes.extend(self.tree_debug.to_bytes()?);
        // bytes.extend(self.ensemble.to_bytes()?);
        // bytes.put_u32_le(self.next_ensemble_offset);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let node = bytes.get_u64_le();
        let depth = bytes.get_u32_le();
        let child_states = if bytes.get_u8() == 1 {
            let n = bytes.get_u64_le() as usize;
            let mut states = HashMap::new();
            for _ in 0..n {
                let i = bytes.get_u64_le();
                let state = SPAState::from_bytes(bytes)?;
                states.insert(i, state);
            }
            Some(states)
        } else {
            None
        };

        let patches = Patches::from_bytes(bytes)?;
        let tree_debug = TreeDebug::from_bytes(bytes)?;
        // let ensemble = Vec::<NodeAndDepth>::from_bytes(bytes)?;
        // let next_ensemble_offset = bytes.get_u32_le();

        Ok(Self {
            node,
            depth,
            child_states,
            patches,
            tree_debug,
            ensemble: Vec::new(),
            next_ensemble_offset: 0,
        })
    }
}

#[derive(Clone)]
pub struct LZ78EnsembleState {
    pub states: Vec<LZ78State>,
    pub base_state: LZ78State,
    pub max_size: u64,
    pub pool: Option<Arc<ThreadPool>>,
}

impl LZ78EnsembleState {
    pub fn new(max_size: u64, base_state: LZ78State, parallel: bool) -> Self {
        let pool = if parallel {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(max_size as usize)
                    .build()
                    .unwrap(),
            ))
        } else {
            None
        };
        Self {
            states: vec![base_state.clone()],
            base_state,
            max_size,
            pool,
        }
    }

    pub fn change_parallelism(&mut self, parallel: bool) -> Result<()> {
        if parallel == self.is_parallel() {
            return Ok(());
        }
        if parallel {
            self.pool = Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(self.max_size as usize)
                    .build()?,
            ));
        } else {
            self.pool = None;
        }

        Ok(())
    }

    pub fn resize(&mut self, max_size: u64) -> Result<()> {
        if self.pool.is_some() {
            self.pool = Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(max_size as usize)
                    .build()?,
            ));
        }
        self.max_size = max_size;
        self.states.truncate(max_size as usize);

        Ok(())
    }

    pub fn is_parallel(&self) -> bool {
        self.pool.is_some()
    }
}

impl ToFromBytes for LZ78EnsembleState {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.max_size);
        bytes.put_u64_le(self.states.len() as u64);
        for state in &self.states {
            bytes.extend(state.to_bytes()?);
        }
        bytes.extend(self.base_state.to_bytes()?);
        bytes.put_u8(self.pool.is_some() as u8);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let max_size = bytes.get_u64_le();
        let n = bytes.get_u64_le();
        let mut states = Vec::with_capacity(n as usize);
        for _ in 0..n {
            states.push(LZ78State::from_bytes(bytes)?);
        }
        let base_state = LZ78State::from_bytes(bytes)?;
        let pool = if bytes.get_u8() > 0 {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(max_size as usize)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            states,
            base_state,
            max_size,
            pool,
        })
    }
}
