use crate::{
    sequence::{Sequence, SequenceSlice},
    storage::ToFromBytes,
};
use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};
use std::collections::HashMap;

/// A node of the LZ78 tree. All nodes in the LZ78 tree are stored as an array,
/// and the tree structure is encoded by storing the index of the child node in
/// the `nodes` vector within the tree root. For instance, consider the sequence
///     00010111,
/// which is parsed into phrases as 0, 00, 1, 01, 11, would have the tree
/// structure:
///```ignore
///                                []
///                           [0]      [1]
///                       [00]  [01]      [11],
/// ```
///
/// and the nodes would be stored in the root of the tree in the same order as
/// the parsed phrases. The root always has index 0, so, in this example, "0"
/// would have index 1, "00" would have index 2, etc.. In that case, the root
/// would have `branch_idxs = {0 -> 1, 1 -> 3}`, the node "0" would have
/// `branch_idxs = {0 -> 2, 1 -> 4}`, and the node "1" would have
/// `branch_idxs = {1 -> 5}`.
#[derive(Debug, Clone)]
pub struct LZ78TreeNode {
    /// Encoding of the tree structure
    pub branch_idxs: HashMap<u32, u64>,
}

impl ToFromBytes for LZ78TreeNode {
    /// Used for saving an LZ78Tree to a file
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.branch_idxs.len() as u64);
        for (&sym, &branch_idx) in self.branch_idxs.iter() {
            bytes.put_u32_le(sym);
            bytes.put_u64_le(branch_idx);
        }

        Ok(bytes)
    }

    /// Use for reading an LZ78Tree from a file
    fn from_bytes(bytes: &mut Bytes) -> Result<Self> {
        let mut branch_idxs: HashMap<u32, u64> = HashMap::new();

        let n_branches = bytes.get_u64_le();
        for _ in 0..n_branches {
            let (sym, branch_idx) = (bytes.get_u32_le(), bytes.get_u64_le());
            branch_idxs.insert(sym, branch_idx);
        }

        Ok(Self { branch_idxs })
    }
}

/// The root of the LZ78 tree. Stores a list of all nodes within the tree, as
/// well as metadata like the SPA parameter and alphabet size. See the
/// documentation of LZ78TreeNode for a detailed description (+example) of the
/// `nodes` array and how the tree structure is encoded.
#[derive(Debug, Clone)]
pub struct LZ78Tree {
    /// List of all nodes in the LZ78 tree, in the order in which they were
    /// parsed
    nodes: Vec<LZ78TreeNode>,
    alphabet_size: u32,
}

/// Returned after traversing the LZ78 tree to a leaf node. Contains all info
/// one may need about the traversal
pub struct LZ78TraversalResult {
    /// The index of the input sequence that corresponds to the end of the
    /// phrase
    pub phrase_prefix_len: u64,
    /// If a leaf was added to the LZ78 tree as a result of the traversal, this
    /// contains the value of the leaf. Otherwise, it is None.
    pub added_leaf: Option<u32>,
    /// The index of the `nodes` array corresponding to the last node
    /// traversed. If a leaf was added to the tree, this is the index of the
    /// leaf's parent, not the leaf itself.
    pub state_idx: u64,
}

impl ToFromBytes for LZ78Tree {
    /// Used for storing an LZ78Tree to a file
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_u64_le(self.nodes.len() as u64);
        for node in self.nodes.iter() {
            bytes.extend(node.to_bytes()?);
        }

        Ok(bytes)
    }

    /// Used for reconstructing an LZ78Tree from a file
    fn from_bytes(bytes: &mut Bytes) -> Result<Self> {
        let alphabet_size = bytes.get_u32_le();
        let n_nodes = bytes.get_u64_le();
        let mut nodes: Vec<LZ78TreeNode> = Vec::with_capacity(n_nodes as usize);

        for _ in 0..n_nodes {
            nodes.push(LZ78TreeNode::from_bytes(bytes)?);
        }

        Ok(Self {
            alphabet_size,
            nodes,
        })
    }
}

impl LZ78Tree {
    pub const ROOT_IDX: u64 = 0;

    /// New LZ78Tree with the default value of the Dirichlet parameter
    /// (i.e., the Jeffreys prior)
    pub fn new(alphabet_size: u32) -> Self {
        let root = LZ78TreeNode {
            branch_idxs: HashMap::new(),
        };

        Self {
            nodes: vec![root],
            alphabet_size,
        }
    }

    pub fn num_phrases(&self) -> u64 {
        self.nodes.len() as u64 - 1
    }

    pub fn is_leaf(&self, idx: u64) -> bool {
        self.get_node(idx).branch_idxs.len() == 0
    }

    /// Get a reference to any node in the LZ78 Tree
    pub fn get_node(&self, idx: u64) -> &LZ78TreeNode {
        &self.nodes[idx as usize]
    }

    /// Get a mutable reference to any node in the LZ78 Tree
    fn get_node_mut(&mut self, idx: u64) -> &mut LZ78TreeNode {
        &mut self.nodes[idx as usize]
    }

    /// Given a node of the tree and a symbol, return the next node in the
    /// traversal. If the start_node does not have a branch corresponding
    /// to sym, this returns the root.
    pub fn traverse_one_symbol(&self, start_node: u64, sym: u32) -> u64 {
        if self.get_node(start_node).branch_idxs.contains_key(&sym) {
            self.get_node(start_node).branch_idxs[&sym]
        } else {
            Self::ROOT_IDX
        }
    }

    /// Returns a tuple of the post-traversal state. If the state is the root,
    /// this means that a new leaf was added.
    pub fn traverse_one_symbol_and_maybe_grow(&mut self, start_node: u64, sym: u32) -> u64 {
        let new_state = self.traverse_one_symbol(start_node, sym);
        if new_state == Self::ROOT_IDX {
            // add a new leaf
            let new_node_idx = self.num_phrases() + 1;
            self.get_node_mut(start_node)
                .branch_idxs
                .insert(sym, new_node_idx);

            self.nodes.push(LZ78TreeNode {
                branch_idxs: HashMap::new(),
            });
        }

        new_state
    }

    /// Start at the root and traverse the tree, using the slice of input
    /// sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true, then the `seen_count` of each traversed
    /// node will be incremented.
    pub fn traverse_root_to_leaf<'a, T>(
        &mut self,
        x: SequenceSlice<'a, T>,
        grow: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence + ?Sized,
    {
        self.traverse_to_leaf_from(Self::ROOT_IDX, x, grow)
    }

    /// Start at a given node of the tree and traverse the tree, using the
    /// slice of input sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true,
    pub fn traverse_to_leaf_from<'a, T>(
        &mut self,
        node_idx: u64,
        x: SequenceSlice<'a, T>,
        grow: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence + ?Sized,
    {
        // keeps track of the current node as we traverse the tree
        let mut state_idx = node_idx;

        // tracks whether a new leaf can be added to the tree
        let mut added_leaf: Option<u32> = None;
        // this will be populated with the index corresponding to the end of
        // the phrase. This is the index of the newly-added leaf, if a leaf is
        // added.
        let mut len = x.len();

        for i in 0..len {
            let val = x.try_get(i)?;
            state_idx = if grow {
                self.traverse_one_symbol_and_maybe_grow(state_idx, val)
            } else {
                self.traverse_one_symbol(state_idx, val)
            };

            if state_idx == Self::ROOT_IDX {
                if grow {
                    added_leaf = Some(val);
                }
                len = i;
                break;
            }
        }

        Ok(LZ78TraversalResult {
            phrase_prefix_len: len,
            state_idx,
            added_leaf,
        })
    }
}
