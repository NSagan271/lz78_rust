use std::collections::HashMap;

use anyhow::{bail, Result};
use bytes::{Buf, BufMut};

use crate::storage::ToFromBytes;

use super::{Ensemble, SPAParams};

pub const LZ_ROOT_IDX: u64 = 0;

#[derive(Clone)]
pub enum SPAState {
    LZ78(LZ78State),
    LZ78Ensemble(LZ78EnsembleState),
    CTW(CTWState),
    None,
}

impl SPAState {
    pub fn get_new_state(params: &SPAParams) -> Self {
        match params {
            SPAParams::LZ78(params) => {
                let root_state = Self::get_new_state(&params.inner_params);
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
                };
                if params.ensemble == Ensemble::None {
                    Self::LZ78(state)
                } else {
                    Self::LZ78Ensemble(LZ78EnsembleState {
                        base_state: state.clone(),
                        states: vec![state],
                        max_size: params.ensemble.get_num_states() as u64,
                    })
                }
            }
            SPAParams::CTW(_) => Self::CTW(CTWState { context: vec![] }),
            _ => Self::None,
        }
    }

    pub fn reset(&mut self) {
        match self {
            SPAState::LZ78(state) => state.reset(),
            SPAState::CTW(state) => {
                state.context.clear();
            }
            SPAState::LZ78Ensemble(state) => {
                state.states.truncate(1);
                state.states[0].reset();
                state.base_state.reset();
            }
            _ => {}
        }
    }

    pub fn try_get_ensemble(&mut self) -> Result<&mut LZ78EnsembleState> {
        match self {
            Self::LZ78Ensemble(state) => Ok(state),
            _ => bail!("Invalid state for LZ78 Ensemble SPA"),
        }
    }

    pub fn try_get_lz78(&mut self) -> Result<&mut LZ78State> {
        match self {
            Self::LZ78(state) => Ok(state),
            Self::LZ78Ensemble(ens) => Ok(&mut ens.base_state),
            _ => bail!("Invalid state for LZ78 SPA"),
        }
    }

    pub fn try_get_ctw(&mut self) -> Result<&mut CTWState> {
        match self {
            Self::CTW(state) => Ok(state),
            _ => bail!("Invalid state for CTW SPA"),
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
            SPAState::CTW(state) => {
                bytes.put_u8(1);
                bytes.extend(state.to_bytes()?);
            }
            SPAState::None => {
                bytes.put_u8(2);
            }
            SPAState::LZ78Ensemble(state) => {
                bytes.put_u8(3);
                bytes.extend(state.to_bytes()?);
            }
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match bytes.get_u8() {
            0 => SPAState::LZ78(LZ78State::from_bytes(bytes)?),
            1 => SPAState::CTW(CTWState::from_bytes(bytes)?),
            2 => SPAState::None,
            3 => SPAState::LZ78Ensemble(LZ78EnsembleState::from_bytes(bytes)?),
            _ => bail!("Invalid SPAState type"),
        })
    }
}

#[derive(Clone)]
pub struct LZ78State {
    pub node: u64,
    pub depth: u32,
    pub child_states: Option<HashMap<u64, SPAState>>,
}

impl LZ78State {
    pub fn get_child_state(&mut self, child_params: &SPAParams) -> Option<&mut SPAState> {
        if let Some(children) = &mut self.child_states {
            if !children.contains_key(&self.node) {
                children.insert(self.node, SPAState::get_new_state(child_params));
            }
            children.get_mut(&self.node)
        } else {
            None
        }
    }

    pub fn go_to_root(&mut self) {
        self.node = LZ_ROOT_IDX;
        self.depth = 0;
    }

    pub fn reset(&mut self) {
        self.go_to_root();
        if let Some(children) = &mut self.child_states {
            children.clear();
        }
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

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let node = bytes.get_u64_le();
        let depth = bytes.get_u32_le();
        let child_states = if bytes.get_u8() == 1 {
            let n = bytes.get_u64_le() as usize;
            let mut states = HashMap::with_capacity(n);
            for _ in 0..n {
                let i = bytes.get_u64_le();
                let state = SPAState::from_bytes(bytes)?;
                states.insert(i, state);
            }
            Some(states)
        } else {
            None
        };

        Ok(Self {
            node,
            depth,
            child_states,
        })
    }
}

#[derive(Clone)]
pub struct LZ78EnsembleState {
    pub states: Vec<LZ78State>,
    pub base_state: LZ78State,
    pub max_size: u64,
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

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
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
        Ok(Self {
            states,
            base_state,
            max_size,
        })
    }
}

#[derive(Clone)]
pub struct CTWState {
    pub context: Vec<u32>,
}

impl ToFromBytes for CTWState {
    fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u32_le(self.context.len() as u32);
        for &sym in self.context.iter() {
            bytes.put_u32_le(sym);
        }
        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let k = bytes.get_u32_le();
        let mut context = Vec::with_capacity(k as usize);
        for _ in 0..k {
            context.push(bytes.get_u32_le());
        }

        Ok(Self { context })
    }
}
