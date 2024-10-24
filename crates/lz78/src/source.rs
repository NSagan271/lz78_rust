// use std::{cell::RefCell, collections::HashMap};

// use anyhow::{bail, Result};
// use rand::Rng;

// use crate::{
//     sequence::{Sequence, U32Sequence},
//     tree::LZ78Tree,
//     util::sample_from_pdf,
// };

// /// Node of an LZ78 probability source
// pub trait SourceNode {
//     /// Creates a new leaf, branching off of the current node
//     fn new_child(&self, rng: &mut impl Rng) -> Self;

//     /// Returns the probability of traversing to each child node
//     fn spa(&self, spa: &LZ78Tree, node_idx: u64) -> Vec<f64>;
// }

// /// Binary LZ78 proabaility source, where each node is associated with a
// /// Bernoulli parameter, Theta. This node generates values i.i.d. Ber(Theta).
// /// New child nodes draw Theta according to a discrete distribution defined
// /// by `theta_pdf` and `theta_values`.
// pub struct DiscreteThetaBinarySourceNode {
//     theta: f64,
//     theta_pmf: Vec<f64>,
//     theta_values: Vec<f64>,
// }

// impl SourceNode for DiscreteThetaBinarySourceNode {
//     fn new_child(&self, rng: &mut impl Rng) -> Self {
//         Self::new(self.theta_pmf.clone(), self.theta_values.clone(), rng)
//     }

//     fn spa(&self, _tree: &LZ78Tree, _node_idx: u64) -> Vec<f64> {
//         // Ber(Theta) PMF
//         vec![1.0 - self.theta, self.theta]
//     }
// }

// impl DiscreteThetaBinarySourceNode {
//     pub fn new(theta_pmf: Vec<f64>, theta_values: Vec<f64>, rng: &mut impl Rng) -> Self {
//         // draw a Bernoulli parameter for the new node
//         let new_theta = theta_values[sample_from_pdf(&theta_pmf, rng.gen_range(0.0..1.0)) as usize];
//         Self {
//             theta: new_theta,
//             theta_pmf,
//             theta_values,
//         }
//     }
// }

// /// Data structure for debugging a LZ78 probability source with
// /// DiscreteThetaBinarySourceNode nodes. Stores the value of Theta (the
// /// Bernoulli parameter) at each time step
// pub struct DiscreteThetaBinarySourceNodeInspector {
//     node: DiscreteThetaBinarySourceNode,
//     /// Value of Theta at each timestep. The datatype RefCell<Vec<f64>> is
//     /// the Rust way of having a pointer to a mutable array shared between all
//     /// nodes
//     theta_list: RefCell<Vec<f64>>,
// }

// impl SourceNode for DiscreteThetaBinarySourceNodeInspector {
//     fn new_child(&self, rng: &mut impl Rng) -> Self {
//         DiscreteThetaBinarySourceNodeInspector {
//             node: self.node.new_child(rng),
//             theta_list: self.theta_list.clone(),
//         }
//     }

//     fn spa(&self, tree: &LZ78Tree, node_idx: u64) -> Vec<f64> {
//         let mut list = self.theta_list.borrow_mut();
//         list.push(self.node.theta);
//         self.node.spa(tree, node_idx)
//     }
// }

// impl DiscreteThetaBinarySourceNodeInspector {
//     pub fn new(
//         theta_list: RefCell<Vec<f64>>,
//         theta_pdf: Vec<f64>,
//         theta_values: Vec<f64>,
//         rng: &mut impl Rng,
//     ) -> Self {
//         let node = DiscreteThetaBinarySourceNode::new(theta_pdf, theta_values, rng);
//         Self { node, theta_list }
//     }
// }

// /// LZ78 probability source node that generates values based on the (Dirichlet-
// /// prior-based) distribution at the given node of the LZ78 tree. This just
// /// builds a regular LZ78 tree and uses the SPA at the current node for
// /// generating from the probability source.
// pub struct DefaultLZ78SourceNode {}

// impl SourceNode for DefaultLZ78SourceNode {
//     fn new_child(&self, _rng: &mut impl Rng) -> Self {
//         Self {}
//     }

//     fn spa(&self, tree: &LZ78Tree, node_idx: u64) -> Vec<f64> {
//         tree.compute_spa(node_idx)
//     }
// }

// /// An LZ78-based probability source, which consists of an LZ78 prefix tree,
// /// where each node has a corresponding SourceNode, which encapsulates how
// /// values are generated from this probability source.
// pub struct LZ78Source<T: SourceNode> {
//     tree: LZ78Tree,
//     /// Maps the index of each LZ78 prefix tree node to the corresponding
//     /// SourceNode
//     tree_node_to_source_node: HashMap<u64, T>,
//     /// Indexes the current node of the LZ78 prefix tree
//     state: u64,
//     alphabet_size: u32,
//     /// Running (un-normalized) log loss incurred so far
//     log_loss: f64,
// }

// impl<T> LZ78Source<T>
// where
//     T: SourceNode,
// {
//     /// Given a SourceNode that is the root of the tree, creates an LZ78
//     /// probability source
//     pub fn new(alphabet_size: u32, source_node: T, gamma: Option<f64>) -> Self {
//         let mut tree_node_to_source_node: HashMap<u64, T> = HashMap::new();
//         tree_node_to_source_node.insert(LZ78Tree::ROOT_IDX, source_node);
//         Self {
//             tree: if let Some(g) = gamma {
//                 LZ78Tree::new_spa(alphabet_size, g)
//             } else {
//                 LZ78Tree::new(alphabet_size)
//             },
//             tree_node_to_source_node,
//             state: LZ78Tree::ROOT_IDX,
//             alphabet_size,
//             log_loss: 0.0,
//         }
//     }

//     /// Generates symbols from the probability source
//     pub fn generate_symbols(&mut self, n: u64, rng: &mut impl Rng) -> Result<U32Sequence> {
//         // output array
//         let mut syms = U32Sequence::new(self.alphabet_size);

//         for i in 0..n {
//             // current node in the LZ78 prefix tree
//             let node = &self.tree_node_to_source_node[&self.state];

//             // generate the next symbol based on the PMF provided by the
//             // current SourceNode
//             let spa = node.spa(&self.tree, self.state);
//             if spa.len() as u32 != self.alphabet_size {
//                 bail!("alphabet size specified incompatible with SourceNode implementation");
//             }
//             let next_sym = sample_from_pdf(&spa, rng.gen_range(0.0..1.0)) as u32;
//             syms.put_sym(next_sym)?;

//             // traverse the LZ78 tree according to the newly-drawn symbol
//             let traverse_result =
//                 self.tree
//                     .traverse_to_leaf_from(self.state, &syms, i, i + 1, true, true)?;
//             self.state = traverse_result.state_idx;
//             self.log_loss += traverse_result.log_loss;

//             // if a new leaf was added to the prefix tree, we also need to create a new SourceNode
//             if let Some(leaf) = traverse_result.added_leaf {
//                 self.tree_node_to_source_node.insert(
//                     self.tree.get_node(self.state).branch_idxs[&leaf],
//                     node.new_child(rng),
//                 );
//                 self.state = LZ78Tree::ROOT_IDX;
//             }
//         }

//         Ok(syms)
//     }
// }

// #[cfg(test)]
// mod tests {

//     use rand::thread_rng;

//     use super::*;

//     #[test]
//     fn test_bernoulli_source() {
//         let mut rng = thread_rng();
//         let mut source = LZ78Source::new(
//             2,
//             DiscreteThetaBinarySourceNode::new(vec![0.5, 0.5], vec![0.0, 1.0], &mut rng),
//             None,
//         );

//         let output = source
//             .generate_symbols(100, &mut rng)
//             .expect("generation failed");

//         let mut i = 0;
//         let mut phrase_num = 0;
//         while i + 2 * phrase_num + 1 < output.len() {
//             assert_eq!(
//                 output.data[i as usize..=(i + phrase_num) as usize],
//                 output.data[(i + phrase_num + 1) as usize..=(i + 2 * phrase_num + 1) as usize]
//             );
//             i += phrase_num + 1;
//             phrase_num += 1;
//         }
//     }

//     #[test]
//     fn sanity_check_lz778_source() {
//         let mut rng = thread_rng();
//         let mut source = LZ78Source::new(4, DefaultLZ78SourceNode {}, None);

//         let output = source
//             .generate_symbols(50, &mut rng)
//             .expect("generation failed");

//         println!("{:?}", output.data);
//     }
// }
