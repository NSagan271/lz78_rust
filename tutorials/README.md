# LZ78 Rust Implementation: Tutorials
This directory contains tutorials and documentation for various aspects of the LZ78 Rust implementation.

## Index
- `SPACodeStructure.md`: documentation of the code related to the LZ78 sequential probability assignment (SPA).
Documentation for other components is pending.
- **`Sequences.ipynb`: prerequisite tutorial for  `CompressorTutorial.ipynb` and `SPATutorial.ipynb`. Look at this first!**

    This tutorial describes the interface used for sequences of integers and characters.
- `SPATutorial.ipynb`: tutorial for use of the LZ78 SPA
- `CompressorTutorial.ipynb`: tutorial for LZ78 compression
- `ProbabilitySourceTutorial.ipynb`: tutorial for an LZ78-based probability source

## Important Note: Python Bindings and Jupyter
Sometimes, Jupyter doesn't register that a cell containing code from the `lz78` library has started running, so it seems like the cell is waiting to run until it finishes. This can be annoying for operations that take a while to run, and **can be remedied by putting `stdout.flush()` at the beginning of the cell**.

## Setup Instructions
You need to install Rust and Maturin, and then install the Python bindings for the `lz78` library as an editable Python package.
1. Install Rust: [Instructions](https://www.rust-lang.org/tools/install).
    - After installing Rust, close and reopen your terminal before proceeding.
2. If applicable, switch to the desired Python environment.
3. Install Maturin: `pip install maturin`
4. Install the `lz78` Python package: `cd crates/python && maturin develop -r && cd ../..`

**NOTE**: If you use virtual environments, you may run into an issue. If you are a conda user, it's possible the `(base)` environment may be activated on startup. `maturin` does not allow for two active virtual environments (ie. via `venv` and `conda`). You must make sure only one is active. One solution is to run `conda deactivate` in preference of your `venv` based virtual environment.

**NOTE**: If you are using MacOS, you may run into the following error with `maturin develop`:
```
error [E0463]: can't find crate for core
    = note: the X86_64-apple-darwin target may not be installed
    = help: consider downloading the target with 'rustup target add x86_64-apple-darwin'
```
Running the recommended command `rustup target add x86_64-apple-darwin` should resolve the issue.

### Notes: Rust Development
If you are modifying the Rust code and are using VSCode, you have to do a few more steps:
1. Install the `rust` and `rust-analyzer` extensions.
2. Adding extra environment variablers to the rust server:
    - In a terminal, run `echo $PATH`, and copy the output.
    - Go to `Preferences: Remote Settings (JSON)` if you are working on a remote machine, or `Preferences: User Settings (JSON)` if you are working locally (you can find this by pressing `F1` and then searching), and make sure it looks like the following:
        ```
        {
            "rust-analyzer.runnables.extraEnv": {
                "PATH": "<the string you copied in the previous step>"
            },
        }
        ```
3. Open `User Settings (JSON)` and add `"editor.formatOnSave": true`
4. Restart your VSCode window.
