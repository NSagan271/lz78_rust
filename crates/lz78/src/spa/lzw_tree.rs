use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};
use hashbrown::{HashMap, HashSet};

use crate::storage::ToFromBytes;

#[derive(Debug, Clone)]
pub struct LZWTree {
    pub branches: HashMap<(u64, u32), u64>,
    pub internal_nodes: HashSet<u64>,
}

unsafe impl Sync for LZWTree {}

impl ToFromBytes for LZWTree {
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.put_u64_le(self.branches.len() as u64);
        for ((k1, k2), v) in self.branches.iter() {
            bytes.put_u64_le(*k1);
            bytes.put_u32_le(*k2);
            bytes.put_u64_le(*v);
        }
        bytes.put_u64_le(self.internal_nodes.len() as u64);
        for k in self.internal_nodes.iter() {
            bytes.put_u64_le(*k);
        }

        Ok(bytes)
    }

    fn from_bytes(bytes: &mut Bytes) -> Result<Self>
    where
        Self: Sized,
    {
        let n = bytes.get_u64_le() as usize;
        let mut branches = HashMap::with_capacity(n);

        for _ in 0..n {
            let (k1, k2, v) = (bytes.get_u64_le(), bytes.get_u32_le(), bytes.get_u64_le());
            branches.insert((k1, k2), v);
        }

        let n = bytes.get_u64_le() as usize;
        let mut internal_nodes = HashSet::with_capacity(n);

        for _ in 0..n {
            let k = bytes.get_u64_le();
            internal_nodes.insert(k);
        }

        Ok(Self {
            branches,
            internal_nodes,
        })
    }
}

impl LZWTree {
    pub fn new() -> Self {
        Self {
            branches: HashMap::new(),
            internal_nodes: HashSet::new(),
        }
    }

    pub fn get_child_idx(&self, idx: u64, sym: u32) -> Option<&u64> {
        self.branches.get(&(idx, sym))
    }

    pub fn add_leaf(&mut self, idx: u64, sym: u32, child_idx: u64) {
        self.branches.insert((idx, sym), child_idx);
        self.internal_nodes.insert(idx);
    }

    pub fn is_leaf(&self, idx: u64) -> bool {
        !self.internal_nodes.contains(&idx)
    }

    pub fn remove_batch(&mut self, nodes: &HashSet<u64>) {
        self.branches = self
            .branches
            .iter()
            .filter(|((parent, _), child)| !nodes.contains(parent) && !nodes.contains(*child))
            .map(|((parent, sym), child)| ((*parent, *sym), *child))
            .collect();
        self.internal_nodes = self
            .branches
            .iter()
            .filter(|((parent, _), child)| !nodes.contains(parent) && !nodes.contains(*child))
            .map(|((parent, _), _)| (*parent))
            .collect();
    }

    pub fn replace(&mut self, node_map: &HashMap<u64, u64>) {
        self.branches = self
            .branches
            .iter()
            .map(|((parent, sym), child)| {
                (
                    (*node_map.get(parent).unwrap_or(parent), *sym),
                    *node_map.get(child).unwrap_or(child),
                )
            })
            .collect();
        self.internal_nodes = self
            .internal_nodes
            .iter()
            .map(|node| *node_map.get(node).unwrap_or(node))
            .collect();
    }

    pub fn shrink_to_fit(&mut self) {
        self.branches.shrink_to_fit();
        self.internal_nodes.shrink_to_fit();
    }
}
