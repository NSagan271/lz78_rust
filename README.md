# LZ78 Sequential Probability Assignment
This code is associated with the paper [A Family of LZ78-based Universal Sequential Probability Assignments](https://arxiv.org/abs/2410.06589).

The codebase is in Rust, with a Python API available. This tutorial goes through how to use the Python API; if you are familiar with Rust or want to learn, feel free to look at `crates/lz78` for the source code and `crates/python` for the bindings (the former is well-documented, whereas the latter is not-so-well-documented).

## Setup
You need to install Rust and Maturin, and then install the Python bindings for the `lz78` library as an editable Python package.
1. Install Rust: [Instructions](https://www.rust-lang.org/tools/install).
    - After you are done installing Rust, restart your terminal.
2. If applicable, switch to the desired Python environment.
3. Install Maturin: `pip install maturin`
4. Install the `lz78` Python package: `cd crates/python && maturin develop && cd ../..`

**NOTE**: If you use virtual environments, you may run into an issue. If you are a conda user, it's possible the `(base)` environment may be activated on startup. `maturin` does not allow for two active virtual environments (ie. via `venv` and `conda`). You must make sure only one is active. One solution is to run `conda deactivate` in preference of your `venv` based virtual environment.

**NOTE**: If you are using MacOS, you may run into the following error with `maturin develop`:
```
error [E0463]: can't find crate for core
    = note: the X86_64-apple-darwin target may not be installed
    = help: consider downloading the target with 'rustup target add ×86_64-apple-darwin'
```
Running the recommended command `rustup target add ×86_64-apple-darwin` should resolve the issue.

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

## Python Interface
See `lz78_python_interface_tutorial.ipynb` for a tutorial on the python API.

## Rust-based Experiments
Experiments performed for the paper are in `crates/experiments`.

There is quite a bit of setup required for the experiments, which is described below for a linux or macOS machine.
Setup may vary for Windows.

### Prerequisites
Make sure you have done everything in the **Setup** section.
You also need to have `git-lfs` ([git large-file storage](https://git-lfs.com/)) installed.

Before downloading datasets, do `mkdir data && cd data` to make and enter the directory where you will store all datasets.

### Huggingface-based datasets:
- **C4**: a huge dataset of text scraped from the internet, and is 13 TB in size.
    We specifically use the the "realnewslike" subset, which is about 15 GB.
    To download only the realnewslike segment, use the following commands:
    ```
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
    cd c4
    git lfs pull --include *realnewslike*
    cd ..
    ```
- **imdb**: a classification dataset of negative and positive movie reviews.
    ```
    git clone https://huggingface.co/datasets/stanfordnlp/imdb
    ```
- **MNIST**: an image classification dataset of handwritten digits, represented as 28x28 binary images.
    ```
    git clone https://huggingface.co/datasets/ylecun/mnist
    ```
- **FashionMNIST**: an image classification dataset of clothing items, represented as 28x28 grayscale images.
    ```
    git clone https://huggingface.co/datasets/zalando-datasets/fashion_mnist    ```
- **wikitext**: text from wikipedia, used for text generation training.
    ```
    git clone https://huggingface.co/datasets/Salesforce/wikitext
    ```
- **TinyStories**: this is a GPT-generated dataset of children's stories over a very small alphabet.
    ```
    git clone https://huggingface.co/datasets/roneneldan/TinyStories
    ```
### Other Datasets
- **Enrom spam**: spam email classification.
    ```
    git clone https://github.com/MWiechmann/enron_spam_data.git
    cd enron_spam_data
    unzip enron_spam_data.zip 
    cd ..
    ```
- **CIFAR10**: image classification of 32x32 RGB images.
    ```
    mkdir cifar
    cd cifar
    curl https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz > cifar10.tar.gz
    tar -xzvf cifar10.tar.gz
    cd ..
    ```
- **TinyShakspeare**: a cleaned version of Shakespeare's plays from Project Gutenberg
    ```
    mkdir shakespeare
    cd shakespeare
    curl https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt > input.txt
    cd ..
    ```

