use std::collections::HashMap;

use anyhow::bail;
use anyhow::Result;

use crate::sequence::Sequence;

/// Returned after traversing the LZ78 tree to a leaf node. Contains all info
/// one may need about the traversal
pub struct LZ78TraversalResult {
    /// If a leaf was added to the LZ78 tree as a result of the traversal, this
    /// contains the value of the leaf. Otherwise, it is None.
    pub added_leaf: Option<u32>,
    /// The index of the `nodes` array corresponding to the last node
    /// traversed. If a leaf was added to the tree, this is the index of the
    /// leaf's parent, not the leaf itself.
    pub state_idx: u64,
}

#[derive(Clone, Debug)]
pub struct LZWData {
    /// A map from (prefix number, symbol) to the number of the phrase
    /// consisting of the prefix and new symbol.
    map: HashMap<(u64, u32), u64>,
    parent_map: Option<HashMap<u64, (u64, u32)>>,
}

impl LZWData {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            parent_map: None,
        }
    }

    pub fn new_with_parent_map() -> Self {
        Self {
            map: HashMap::new(),
            parent_map: Some(HashMap::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Start at the root and traverse the tree, using the slice of input
    /// sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true, then the `seen_count` of each traversed
    /// node will be incremented.
    pub fn traverse_root_to_leaf(
        &mut self,
        input: &mut impl Iterator<Item = u32>,
        pretrained_data: Option<&LZWData>,
    ) -> LZ78TraversalResult {
        self.traverse_to_leaf_from(0, input, pretrained_data)
    }

    /// Start at a given node of the tree and traverse the tree, using the
    /// slice of input sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true,
    pub fn traverse_to_leaf_from(
        &mut self,
        node_idx: u64,
        input: &mut impl Iterator<Item = u32>,
        pretrained_data: Option<&LZWData>,
    ) -> LZ78TraversalResult {
        // keeps track of the current node as we traverse the tree
        let mut state_idx = node_idx;
        let mut added_leaf: Option<u32> = None;
        let mut traversing_pretrained = pretrained_data.is_some();
        let mut current_tree = pretrained_data.unwrap_or(self);
        let pretrain_len = if traversing_pretrained {
            current_tree.len() as u64
        } else {
            0
        };

        for sym in input {
            if current_tree.map.contains_key(&(state_idx, sym)) {
                state_idx = current_tree.map[&(state_idx, sym)];
                continue;
            }
            if traversing_pretrained {
                traversing_pretrained = false;
                current_tree = self;
            }

            if current_tree.map.contains_key(&(state_idx, sym)) {
                state_idx = current_tree.map[&(state_idx, sym)];
            } else {
                let new_node = self.map.len() as u64 + 1 + pretrain_len;
                self.map.insert((state_idx, sym), new_node);
                if let Some(parent_map) = &mut self.parent_map {
                    parent_map.insert(new_node, (state_idx, sym));
                }
                added_leaf = Some(sym);
                break;
            }
        }

        LZ78TraversalResult {
            state_idx,
            added_leaf,
        }
    }

    pub fn get_phrase<T>(
        &self,
        node: u64,
        output_seq: &mut T,
        max_output_len: Option<u64>,
    ) -> Result<()>
    where
        T: Sequence,
    {
        if self.parent_map.is_none() {
            bail!("get_phrase only works if LZWData was intitialized with LZWData::new_with_parent_map()");
        }
        let parent_map = self.parent_map.as_ref().unwrap();
        let mut node = node;
        let mut sym;
        let mut symbols = vec![];
        while parent_map.contains_key(&node) {
            (node, sym) = parent_map[&node];
            symbols.push(sym);
        }

        let max_len = if let Some(x) = max_output_len {
            x - output_seq.len()
        } else {
            symbols.len() as u64
        };

        for &sym in symbols.iter().rev().take(max_len as usize) {
            output_seq.put_sym(sym)?;
        }

        Ok(())
    }
}
