use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use bytes::{Buf, BufMut};

use crate::storage::ToFromBytes;

use super::SPAParams;

#[derive(Clone)]
pub enum SPAState {
    LZ78(LZ78State),
    LZ78Gen(LZ78GenerationState),
    CTW(CTWState),
    None,
}

impl SPAState {
    pub fn get_new_state(params: &SPAParams) -> Self {
        match params {
            SPAParams::LZ78(params) => {
                let root_state = Self::get_new_state(&params.inner_params);
                Self::LZ78(LZ78State {
                    node: 0,
                    child_states: if let Self::None = root_state {
                        None
                    } else {
                        let mut map = HashMap::new();
                        map.insert(0, root_state);
                        Some(map)
                    },
                })
            }
            SPAParams::CTW(_) => Self::CTW(CTWState { context: vec![] }),
            _ => Self::None,
        }
    }

    pub fn get_new_gen_state(params: &SPAParams) -> Self {
        match params {
            SPAParams::LZ78(params) => {
                let root_state = Self::get_new_gen_state(&params.inner_params);
                Self::LZ78Gen(LZ78GenerationState {
                    node: 0,
                    child_states: if let Self::None = root_state {
                        None
                    } else {
                        Some(vec![root_state])
                    },
                    nodes_seen: HashSet::new(),
                    reseeding_seq: vec![],
                    last_time_root_seen: 0,
                })
            }
            _ => Self::get_new_state(params),
        }
    }

    pub fn reset(&mut self) {
        match self {
            SPAState::LZ78(state) => {
                state.node = 0;
                if let Some(children) = &mut state.child_states {
                    children.clear();
                }
            }
            SPAState::CTW(state) => {
                state.context.clear();
            }
            _ => {}
        }
    }

    pub fn try_get_lz78(&mut self) -> Result<&mut LZ78State> {
        match self {
            Self::LZ78(state) => Ok(state),
            _ => bail!("Invalid state for LZ78 SPA"),
        }
    }

    pub fn try_get_ctw(&mut self) -> Result<&mut CTWState> {
        match self {
            Self::CTW(state) => Ok(state),
            _ => bail!("Invalid state for CTW SPA"),
        }
    }

    pub fn try_get_ctw_immut(&self) -> Result<&CTWState> {
        match self {
            Self::CTW(state) => Ok(state),
            _ => bail!("Invalid state for CTW SPA"),
        }
    }
}

#[derive(Clone)]
pub struct LZ78State {
    pub node: u64,
    pub child_states: Option<HashMap<u64, SPAState>>,
}

#[derive(Clone)]

pub struct LZ78GenerationState {
    pub node: u64,
    pub child_states: Option<Vec<SPAState>>,
    pub nodes_seen: HashSet<u64>,
    pub reseeding_seq: Vec<u32>,
    pub last_time_root_seen: u64,
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
