# Sequential Probability Assignment (SPA) Code Structure

## 1. Definitions
### 1.1 Rust Language Features
The below is an abbreviated description of some Rust features that are referenced in this document.
For more information, see [the Rust Book](https://doc.rust-lang.org/book/).
- **Struct**: a struct is an object that contains data and has associated methods.
    The type of data stored in a strict is specified when a struct is defined, e.g.,
    ```
    pub struct StructName {
        an_int: u64,
        float_array: Vec<f64>
    }
    ```
    Methods are specified within `impl` blocks, e.g.,
    ```
    impl StructName {
        fn method_name(&self) {
            ...
        }
    }
    ```
- **Trait**: a trait (mostly equivalent to an Interface in Java, or Abstract Base Class in Python) specifies a list of functions that all structs implementing the trait must have.
    A trait is defined, e.g.,
    ```
    pub trait TraitName {
        fn my_function(&self, input: u64) -> u64;
    }
    ```
    All structs implementing this trait must have the following:
    ```
    impl TraitName for StructName {
        fn my_function(&self, input: u64) -> u64 {
            ...
        }
    }
    ```

### 1.2 SPAs
This is a brief description of some key theoretical features; see the paper [A Family of LZ78-based Universal Sequential Probability Assignments](https://arxiv.org/abs/2410.06589) for more details.
- **Sequence**: a sequence is a list of **symbols** in an alphabet, denoted $x^n$, where $n$ is the length of the sequence, and each element $x_i$ is the member of some finite alphabet $\mathcal{A}$.
    For instance, a binary sequence, $x^n$ , is a length-$n$ bitstream, and $\mathcal{A} = \{0, 1\}$.
- **SPA**: a sequential probability assignment is a type of probability model for individual sequences.
    Specifically, it is a mapping from a prefix of a sequence, $x^{t-1}$, to a probability mass function for the next symbol, $x_t$.
- **Dirichlet mixture SPA**: A Dirichlet mixture SPA is one that assigns
    $$\text{Pr}(x_t = a | x^{t-1}) = \frac{(\text{number of ``$a$''s in $x^{t-1}$}) + \gamma}{t-1 + \gamma|\mathcal{A}|}.$$
    The "Dirichlet smoothing" parameter is referred to in the code and documentation as `gamma`.
- **LZ78 SPA**: For the purpose of this codebase, the LZ78 SPA is one that constructs an LZ78 prefix tree from the training data, and constructs a Dirichlet mixture SPA based on the symbols seen while traversing each node of the tree.

## 2. File Overview

The main Rust code is in `crates/lz78/src`, and the Python interface is in `crates/python`.

In `crates/lz78/src`, the following files are relevant:
- `sequence.rs`: the inputs to the LZ78 SPA all have to be formatted as objects that follow the `Sequence` intergace ("trait"). This allows them to take sequences of different datatypes, including strings, bytes, and integers.
- `spa/` folder: the main code for the LZ78 SPA (and other SPAs, including a Dirichlet mixture SPA).
    - `spa/mod.rs`: this is the "module" file for SPA-related code, which indexes sub-modules and includes structures and traits that are used throughout the LZ78 SPA implementation.
    - `spa/config.rs`: configuration option objects for different types of SPAs (e.g., the `gamma` or smoothing parameter of the Dirichlet SPA). These options are stored in configuration objects and passed into SPA methods during training, inference, etc.
    - `spa/states.rs`: state objects for different types of SPAs. State objects store information like the current node in an LZ78 tree, and are also optionally used for storing some debugging information (e.g., the depths that leaves were added during training, or demarcations of LZ78 phrases during inference). State objectd are also passed into SPA methods during training, inference, etc.
    - `spa/lzw_tree.rs`: the main data structure for storing the branch structure of the LZ78 tree. Each node of the tree is assigned a unique ID (in the order that it was added to the tree), and branches are represented as a hashmap mapping (parent ID, symbol) to the ID of the child node.
    - `spa/dirichlet.rs`: code for the Dirichlet SPA, which is the SPA at every node of the LZ78 tree.
    -`spa/lz_transform.rs`: this file has the main code for the LZ78 SPA (training/inference/generation), including features like ensemble inference and backshift parsing, e.g.
    - `spa/generation.rs`: scaffolding for autogregressive sequence generation.
    - `spa/utils.rs`: helper methods for inference/generation features like temperature, top-k sampling, lower bound, and adaptive gamma.
    

The following utility files may also be relevant:
- `storage.rs`: this contains the trait for converting objects (e.g., SPAs) to binary files.
- `util.rs`: currently just contains a helper function for sampling from a PDF on the integers $\{0, 1, ..., N-1\}$.

In `crates/python`, the following files are relevant:
- `lz78.pyi`: a list of all of the classes and methods available in the python interface, along with docstrings. This is needed, e.g., for documentation and autocomplete features in IDEs. 
- `src/sequence.rs`: Python interfaces for individual sequences of bytes, integers, and strings. These `Sequence` objects are inputs to SPA methods.
- `src/spa.rs`: Python interfaces the LZ78 SPA (with a Dirichlet prior).
- `src/lib.rs`: actual code for adding the LZ78 SPA and related functions to the Python package.

## 3. Struct-Level Documentation
(In progress...)