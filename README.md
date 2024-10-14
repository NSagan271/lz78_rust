# LZ78 Sequential Probability Assignment
This code is associated with the paper [A Family of LZ78-based Universal Sequential Probability Assignments](https://arxiv.org/abs/2410.06589).

The codebase is in Rust, with a Python API available. This tutorial goes through how to use the Python API; if you are familiar with Rust or want to learn, feel free to look at `crates/lz78` for the source code and `crates/python` for the bindings (the former is well-documented, whereas the latter is not-so-well-documented).

## Setup
You need to install Rust and Maturin, and then install the Python bindings for the `lz78` library as an editable Python package.
1. Install Rust: [Instructions](https://www.rust-lang.org/tools/install).
2. If applicable, switch to the desired Python environment.
3. Install Maturin: `pip install maturin`
4. Install the `lz78` Python package: `cd crates/python && maturin develop && cd ../..`

## Python Interface
See `lz78_python_interface_tutorial.ipynb` for a tutorial on the python API.

## Rust-based Experiments
Experiments performed for the paper are in `crates/experiments`. Documentation for these experiments, including instructions on how to download all of the data, is pending.