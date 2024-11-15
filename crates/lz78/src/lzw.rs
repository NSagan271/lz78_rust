use std::collections::HashMap;

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
}

impl LZWData {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
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
    ) -> LZ78TraversalResult {
        self.traverse_to_leaf_from(0, input)
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
    ) -> LZ78TraversalResult {
        // keeps track of the current node as we traverse the tree
        let mut state_idx = node_idx;
        let mut added_leaf = None;

        for sym in input {
            if self.map.contains_key(&(state_idx, sym)) {
                state_idx = self.map[&(state_idx, sym)];
            } else {
                self.map.insert((state_idx, sym), self.map.len() as u64 + 1);
                added_leaf = Some(sym);
                break;
            }
        }

        LZ78TraversalResult {
            state_idx,
            added_leaf,
        }
    }
}
