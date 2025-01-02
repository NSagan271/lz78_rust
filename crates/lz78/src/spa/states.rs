use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use bytes::{Buf, BufMut};

use crate::storage::ToFromBytes;

use super::SPAParams;

#[derive(Clone)]
pub enum SPAState {
    LZ78(LZ78State),
    CTW(CTWState),
    None,
}

impl SPAState {
    pub fn get_new_state(params: &SPAParams, generation: bool) -> Self {
        match params {
            SPAParams::LZ78(params) => {
                let gen_state = if generation {
                    Some(LZ78GenerationState {
                        nodes_seen: HashSet::new(),
                        reseeding_seq: vec![],
                        last_time_root_seen: 0,
                    })
                } else {
                    None
                };

                let root_state = Self::get_new_state(&params.inner_params, generation);
                let child_states = if let Self::None = root_state {
                    None
                } else {
                    let mut map = HashMap::new();
                    map.insert(0, root_state);
                    Some(map)
                };

                Self::LZ78(LZ78State {
                    node: 0,
                    child_states,
                    gen_state,
                })
            }
            SPAParams::CTW(_) => Self::CTW(CTWState { context: vec![] }),
            _ => Self::None,
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
            _ => bail!("Invalid SPAState type"),
        })
    }
}

#[derive(Clone)]
pub struct LZ78State {
    pub node: u64,
    pub child_states: Option<HashMap<u64, SPAState>>,
    pub gen_state: Option<LZ78GenerationState>,
}

impl LZ78State {
    pub fn get_child_state(&mut self, child_params: &SPAParams) -> Option<&mut SPAState> {
        if let Some(children) = &mut self.child_states {
            if !children.contains_key(&self.node) {
                children.insert(
                    self.node,
                    SPAState::get_new_state(child_params, self.gen_state.is_some()),
                );
            }
            children.get_mut(&self.node)
        } else {
            None
        }
    }

    pub fn try_update_gen_state_with_curr_node(&mut self) -> Result<()> {
        let gen_state = self
            .gen_state
            .as_mut()
            .ok_or_else(|| anyhow!("expected generation state"))?;

        if self.node == 0 {
            gen_state.last_time_root_seen = gen_state.reseeding_seq.len() as u64;
        }

        gen_state.nodes_seen.insert(self.node);

        Ok(())
    }

    /// Used by the causally processed SPA
    pub fn try_update_gen_state_with_sym(&mut self, sym: u32) -> Result<()> {
        let gen_state = self
            .gen_state
            .as_mut()
            .ok_or_else(|| anyhow!("expected generation state"))?;
        gen_state.reseeding_seq.push(sym);

        Ok(())
    }
}

impl ToFromBytes for LZ78State {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.node);
        bytes.put_u8(self.child_states.is_some() as u8);
        if let Some(child_states) = &self.child_states {
            bytes.put_u64_le(child_states.len() as u64);
            for (&node, state) in child_states {
                bytes.put_u64_le(node);
                bytes.extend(state.to_bytes()?);
            }
        }

        bytes.put_u8(self.gen_state.is_some() as u8);
        if let Some(gen_state) = &self.gen_state {
            bytes.extend(gen_state.to_bytes()?);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let node = bytes.get_u64_le();
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

        let gen_state = if bytes.get_u8() == 1 {
            Some(LZ78GenerationState::from_bytes(bytes)?)
        } else {
            None
        };

        Ok(Self {
            node,
            child_states,
            gen_state,
        })
    }
}

#[derive(Clone)]

pub struct LZ78GenerationState {
    pub nodes_seen: HashSet<u64>,
    pub reseeding_seq: Vec<u32>,
    pub last_time_root_seen: u64,
}

impl ToFromBytes for LZ78GenerationState {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.nodes_seen.len() as u64);
        self.nodes_seen.iter().for_each(|&n| bytes.put_u64_le(n));

        bytes.put_u64_le(self.reseeding_seq.len() as u64);
        self.reseeding_seq.iter().for_each(|&n| bytes.put_u32_le(n));

        bytes.put_u64_le(self.last_time_root_seen);

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut bytes::Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le() as usize;
        let mut nodes_seen = HashSet::with_capacity(n);
        for _ in 0..n {
            nodes_seen.insert(bytes.get_u64_le());
        }

        let n = bytes.get_u64_le() as usize;
        let mut reseeding_seq = Vec::with_capacity(n);
        for _ in 0..n {
            reseeding_seq.push(bytes.get_u32_le());
        }

        let last_time_root_seen = bytes.get_u64_le();

        Ok(Self {
            nodes_seen,
            reseeding_seq,
            last_time_root_seen,
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
