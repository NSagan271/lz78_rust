from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm
import os
from lz_embed.classical import AlphabetInfo, BasicNGramSpectrum
import matplotlib.pyplot as plt
import mteb
from sys import stdout
import numpy as np
import torch


# NUM_DOC = 500_000
NUM_DOC = 400_000
DIM = 1024
COMPUTE_DEVICE = "cuda:6"
OUTPUT_DIR = "data/ngram-pca-tmp"

def main():
    model = BasicNGramSpectrum(
        alpha_info=AlphabetInfo(valid_character_string="abcdefghijklmnopqrstuvwxyz"),
        n=2, lowercase=True
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(f"{OUTPUT_DIR}/mtx.pkl"):
        print(f"Loading in matrix from {OUTPUT_DIR}/mtx.pkl")
        mtx = torch.load(f"{OUTPUT_DIR}/mtx.pkl", map_location="cpu")[:NUM_DOC, :].to(COMPUTE_DEVICE)
        if COMPUTE_DEVICE != "cpu":
            mtx = mtx.to(torch.float32)
    else:
        mtx = torch.zeros((NUM_DOC, model.fixed_len), dtype=torch.float16)
        data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", limit=NUM_DOC) 
        for (i, document) in enumerate(tqdm(data_reader())):
            mtx[i, :] = model.encode_single(document.text).to(mtx.dtype)

        print(f"Saving matrix to {OUTPUT_DIR}/mtx.pkl")
        torch.save(mtx, f=f"{OUTPUT_DIR}/mtx.pkl")
        print("Done saving")

        mtx = mtx.to(COMPUTE_DEVICE)
        if COMPUTE_DEVICE != "cpu":
            mtx = mtx.to(torch.float32)

    print(f"\nStarting PCA on device {COMPUTE_DEVICE}")
    mtx -= mtx.mean(dim=0) # subtract the mean of each feature
    _, _, subspace = torch.svd_lowrank(mtx, q=DIM*3)
    subspace = subspace[:, :DIM]

    print(f"Saving subspace to {OUTPUT_DIR}/subspace_{DIM}.pkl")
    torch.save(subspace.to("cpu"), f=f"{OUTPUT_DIR}/subspace_{DIM}.pkl")
    print("Done saving")


if __name__ == "__main__":
    main()