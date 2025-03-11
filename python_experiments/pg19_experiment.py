from torch.utils.data import Dataset
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import List
import numpy as np
from datasets import load_dataset
import regex as re
from tqdm import tqdm
from sys import stdout

from lz78 import Sequence, CharacterMap, BlockLZ78Encoder, LZ78SPA
from lz78 import encoded_sequence_from_bytes, spa_from_bytes
from os import makedirs

tf.config.set_visible_devices([], 'GPU')

class PG19DataLoader:
    def __init__(self, data_type: str, start_index: int = 0, batch_size: int = 1, normalize: str = 'none'):
        self.data =  tfds.load('pg19', split=data_type, shuffle_files=False)
        self.dataset = (self.data
                        .skip(start_index)
                        .batch(batch_size)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        print(data_type, ": ", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for batch in self.dataset:
            text_bytes = np.frombuffer(batch['book_text'].numpy()[0], dtype=np.uint8)
            text_bytes = text_bytes.tolist()
            yield text_bytes

train_dataloader = PG19DataLoader("train")
val_dataloader = PG19DataLoader("validation")

model = LZ78SPA(alphabet_size=256,compute_training_loss=False)
model.set_inference_params(ensemble_type="depth", ensemble_n=6, backshift_ctx_len=10, backshift_min_count=1, lb=1e-5, temp=1, gamma=1/256)

stdout.flush()
total_byte = 0
for trn_iter, batch in enumerate(tqdm(train_dataloader, desc="Building LZ tree"), start=1):
    # build LZ model only 1 epoch
    stdout.flush()
    model.train_on_block(Sequence(batch, alphabet_size=256))
    model.reset_state()
    total_byte += len(batch)

    if trn_iter % 5000 == 0:
        print("Total bytes: ", total_byte, "; Training log loss: ", model.get_normalized_log_loss())
        # print("Running inference")
        # val_loss = 0
        # val_byte = 0

        # for inf_batch in tqdm(val_dataloader, desc="Validation"):
        #     stdout.flush()
        #     val_loss += model.compute_test_loss(Sequence(inf_batch, alphabet_size=256))
        #     model.reset_state()
        #     val_byte += len(inf_batch)
        # print("Inference log loss: ", val_loss / val_byte)

        stdout.flush()
        bytes = model.to_bytes()

        makedirs("spa_outputs", exist_ok=True)
        with open("spa_outputs/pg19_spa_not_pruned.bin", 'wb') as file:
            file.write(bytes)
# model.prune(1)
bytes = model.to_bytes()
with open("spa_outputs/pg19_spa_not_pruned.bin", 'wb') as file:
    file.write(bytes)