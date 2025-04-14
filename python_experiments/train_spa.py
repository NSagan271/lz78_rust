from datatrove.pipeline.readers import ParquetReader
from lz_embed.transformer_based import TokenizedLZPlusEmbedding, WeightType
from tqdm import tqdm
import os


MAX_SIZE = 15 * 1e9
NUM_DOC = 5_000_000
SAVE_FREQ = 10_000

def main():
    model = TokenizedLZPlusEmbedding( 
        inner_model_name="BAAI/bge-base-en-v1.5",
        output_dir="data/object2",
        compute_device="cuda:7",
        weight_type=WeightType.LOG_LOSS,
        pca_dim=256,
        pca=True,
        overwrite_objects=True
    )

    data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", limit=NUM_DOC) 
    for (i, document) in enumerate(tqdm(data_reader())):
        # do something with document
        model.spa.reset_state()
        save = (i % SAVE_FREQ) == SAVE_FREQ - 1
        model.train_spa(document.text, save=save)

        if save and os.path.getsize(model.spa_file) > MAX_SIZE:
            print("SPA reached max size! Exiting train loop.")
            break
    model.spa.prune(2)
    model.spa.to_file(model.spa_file)

if __name__ == "__main__":
    main()

