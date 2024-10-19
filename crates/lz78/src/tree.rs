use crate::sequence::Sequence;
use anyhow::Result;
use bytes::{Buf, BufMut, Bytes};
use std::collections::HashMap;

/// A node of the LZ78 tree. All nodes in the LZ78 tree are stored as an array,
/// and the tree structure is encoded by storing the index of the child node in
/// the `nodes` vector within the tree root. For instance, consider the sequence
///     00010111,
/// which is parsed into phrases as 0, 00, 1, 01, 11, would have the tree
/// structure:
///
///                                []
///                           [0]      [1]
///                       [00]  [01]      [11],
///
/// and the nodes would be stored in the root of the tree in the same order as
/// the parsed phrases. The root always has index 0, so, in this example, "0"
/// would have index 1, "00" would have index 2, etc.. In that case, the root
/// would have `branch_idxs = {0 -> 1, 1 -> 3}`, the node "0" would have
/// `branch_idxs = {0 -> 2, 1 -> 4}`, and the node "1" would have
/// `branch_idxs = {1 -> 5}`.
#[derive(Debug, Clone)]
pub struct LZ78TreeNode {
    /// The number of times this node has been visited. Used for computing the
    /// sequential probability assignment
    pub seen_count: u64,
    /// Encoding of the tree structure
    pub branch_idxs: HashMap<u32, u64>,
}

impl LZ78TreeNode {
    /// Used for saving an LZ78Tree to a file
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u64_le(self.seen_count);
        bytes.put_u64_le(self.branch_idxs.len() as u64);
        for (&sym, &branch_idx) in self.branch_idxs.iter() {
            bytes.put_u32_le(sym);
            bytes.put_u64_le(branch_idx);
        }

        bytes
    }

    /// Use for reading an LZ78Tree from a file
    pub fn from_bytes(bytes: &mut Bytes) -> Self {
        let mut branch_idxs: HashMap<u32, u64> = HashMap::new();
        let seen_count = bytes.get_u64_le();

        let n_branches = bytes.get_u64_le();
        for _ in 0..n_branches {
            let (sym, branch_idx) = (bytes.get_u32_le(), bytes.get_u64_le());
            branch_idxs.insert(sym, branch_idx);
        }

        Self {
            seen_count,
            branch_idxs,
        }
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
    /// Dirichlet parameter for if this tree is used as a sequential
    /// probability assignment
    spa_gamma: f64,
    alphabet_size: u32,
}

/// Returned after traversing the LZ78 tree to a leaf node. Contains all info
/// one may need about the traversal
pub struct LZ78TraversalResult {
    /// The index of the input sequence that corresponds to the end of the
    /// phrase
    pub phrase_end_idx: u64,
    /// If a leaf was added to the LZ78 tree as a result of the traversal, this
    /// contains the value of the leaf. Otherwise, it is None.
    pub added_leaf: Option<u32>,
    /// The index of the `nodes` array corresponding to the last node
    /// traversed. If a leaf was added to the tree, this is the index of the
    /// leaf's parent, not the leaf itself.
    pub state_idx: u64,
    /// Self-entropy log loss incurred during this traversal
    pub log_loss: f64,
}

impl LZ78Tree {
    pub const ROOT_IDX: u64 = 0;

    /// New LZ78Tree with the default value of the Dirichlet parameter
    /// (i.e., the Jeffreys prior)
    pub fn new(alphabet_size: u32) -> Self {
        let root = LZ78TreeNode {
            seen_count: 1,
            branch_idxs: HashMap::new(),
        };

        Self {
            nodes: vec![root],
            spa_gamma: 0.5,
            alphabet_size,
        }
    }

    /// New LZ78Tree, specifying the Dirichlet parameter
    pub fn new_spa(alphabet_size: u32, spa_gamma: f64) -> Self {
        let root = LZ78TreeNode {
            seen_count: 1,
            branch_idxs: HashMap::new(),
        };

        Self {
            nodes: vec![root],
            spa_gamma,
            alphabet_size,
        }
    }

    /// Used for storing an LZ78Tree to a file
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.alphabet_size);
        bytes.put_f64_le(self.spa_gamma);
        bytes.put_u64_le(self.nodes.len() as u64);
        for node in self.nodes.iter() {
            bytes.extend(node.to_bytes());
        }

        bytes
    }

    /// Used for reconstructing an LZ78Tree from a file
    pub fn from_bytes(bytes: &mut Bytes) -> Self {
        let alphabet_size = bytes.get_u32_le();
        let spa_gamma = bytes.get_f64_le();
        let n_nodes = bytes.get_u64_le();
        let mut nodes: Vec<LZ78TreeNode> = Vec::with_capacity(n_nodes as usize);

        for _ in 0..n_nodes {
            nodes.push(LZ78TreeNode::from_bytes(bytes));
        }

        Self {
            alphabet_size,
            spa_gamma,
            nodes,
        }
    }

    /// Given a node of the tree, compute the value of the SPA for each
    /// value in the alphabet
    pub fn compute_spa(&self, current_state: u64) -> Vec<f64> {
        let mut spa: Vec<f64> =
            vec![self.compute_spa_val(current_state, None)].repeat(self.alphabet_size as usize);

        for (sym, child_idx) in self.get_node(current_state).branch_idxs.iter() {
            spa[*sym as usize] = self.compute_spa_val(current_state, Some(*child_idx));
        }
        spa
    }

    /// Given a node of the tree and a child (the node corresponding to a
    /// particular "next symbol" in the alphabet), compute the SPA P(a|x^t).
    pub fn compute_spa_val(&self, current_state: u64, next_state: Option<u64>) -> f64 {
        // Number of times we have seen the `next_state` node.
        let next_state_count = if let Some(state) = next_state {
            self.get_node(state).seen_count
        } else {
            0
        } as f64;

        // Dirichlet mixture
        (next_state_count + self.spa_gamma)
            / (self.get_node(current_state).seen_count as f64 - 1.
                + self.spa_gamma * self.alphabet_size as f64)
    }

    /// Compute the log loss incurred from traversing the tree from
    /// `current_state` to `next_state`
    pub fn compute_instantaneous_log_loss(
        &self,
        current_state: u64,
        next_state: Option<u64>,
    ) -> f64 {
        let spa = self.compute_spa_val(current_state, next_state);
        (1.0 / spa).log2()
    }

    pub fn num_phrases(&self) -> u64 {
        self.nodes.len() as u64 - 1
    }

    pub fn is_leaf(&self, idx: u64) -> bool {
        self.get_node(idx).seen_count == 1
    }

    /// Get a reference to any node in the LZ78 Tree
    pub fn get_node(&self, idx: u64) -> &LZ78TreeNode {
        &self.nodes[idx as usize]
    }

    /// Get a mutable reference to any node in the LZ78 Tree
    fn get_node_mut(&mut self, idx: u64) -> &mut LZ78TreeNode {
        &mut self.nodes[idx as usize]
    }

    /// Start at the root and traverse the tree, using the slice of input
    /// sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true, then the `seen_count` of each traversed
    /// node will be incremented.
    pub fn traverse_root_to_leaf<T>(
        &mut self,
        x: &T,
        start_idx: u64,
        end_idx: u64,
        grow: bool,
        update_counts: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence,
    {
        self.traverse_to_leaf_from(Self::ROOT_IDX, x, start_idx, end_idx, grow, update_counts)
    }

    /// Start at a given node of the tree and traverse the tree, using the
    /// slice of input sequence `x` between `start_idx` and `end_idx`.
    ///
    /// If `grow` is true, a leaf will be added to the tree if possible.
    /// If `update_counts` is true,
    pub fn traverse_to_leaf_from<T: ?Sized>(
        &mut self,
        node_idx: u64,
        x: &T,
        start_idx: u64,
        end_idx: u64,
        grow: bool,
        update_counts: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence,
    {
        let num_phrases = self.num_phrases();

        // keeps track of the current node as we traverse the tree
        let mut state_idx = node_idx;

        // tracks whether a new leaf can be added to the tree
        let mut new_leaf: Option<u32> = None;
        // this will be populated with the index corresponding to the end of
        // the phrase. This is the index of the newly-added leaf, if a leaf is
        // added.
        let mut end_idx = end_idx;

        let mut log_loss: f64 = 0.0;

        for i in start_idx..end_idx {
            let val = x.try_get(i)?;

            // state, before traversing the tree with the new symbol
            let prev_state_idx = state_idx;

            if self.get_node(state_idx).branch_idxs.contains_key(&val) {
                // we're not yet at the leaf, so we traverse further
                state_idx = self.get_node(state_idx).branch_idxs[&val];

                log_loss += self.compute_instantaneous_log_loss(prev_state_idx, Some(state_idx));
            } else {
                // we reached the end of a phrase, so we stop traversing and
                // maybe add a new leaf
                new_leaf = Some(val);
                end_idx = i;
                log_loss += self.compute_instantaneous_log_loss(prev_state_idx, None);
            }

            if update_counts {
                self.get_node_mut(prev_state_idx).seen_count += 1;
            }
            if let Some(_) = new_leaf {
                break;
            }
        }

        let added_leaf = if grow { new_leaf } else { None };
        if added_leaf.is_some() {
            // add a new leaf
            self.get_node_mut(state_idx)
                .branch_idxs
                .insert(new_leaf.unwrap(), num_phrases + 1);

            self.nodes.push(LZ78TreeNode {
                seen_count: 1,
                branch_idxs: HashMap::new(),
            });
        }

        Ok(LZ78TraversalResult {
            phrase_end_idx: end_idx,
            state_idx,
            added_leaf,
            log_loss,
        })
    }
}
