from torch.utils.data import Dataset
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sys import stdout

from lz78 import Sequence, LZ78SPA
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

model = LZ78SPA(alphabet_size=256, gamma=1/256, compute_training_loss=False)
model.set_inference_config(
    lb=1e-5,
    temp=1,
    lb_or_temp_first="lb_first",
    ensemble_type="depth",
    ensemble_n=6,
    adaptive_gamma="disabled",
    backshift_parsing=True,
    backshift_ctx_len=10,
    backshift_break_at_phrase=True
)

stdout.flush()
total_byte = 0
for trn_iter, batch in enumerate(tqdm(train_dataloader, desc="Building LZ tree"), start=1):
    # build LZ model only 1 epoch
    stdout.flush()
    model.train_on_block(Sequence(batch, alphabet_size=256))
    model.reset_state()
    total_byte += len(batch)

    if trn_iter % 5000 == 0:
        print("Running inference")

        val_loss = 0
        val_seq = 0
        for inf_batch in tqdm(val_dataloader, desc="Validation"):
            stdout.flush()
            test_seqs = []
            for i in range(0, len(inf_batch)-1023, 512):
                test_seqs.append(inf_batch[i:i+1024])

            inputs = [Sequence(seq[512:],alphabet_size=256) for seq in test_seqs]
            ctxs = [Sequence(seq[:512],alphabet_size=256) for seq in test_seqs]

            res = model.compute_test_loss_parallel(
                inputs, ctxs, num_threads=32
            )

            val_loss += np.sum(np.array([x["avg_log_loss"] for x in res]))
            val_seq += len(test_seqs)
            
        print("Inference log loss: ", val_loss / val_seq)
        stdout.flush()
        makedirs("spa_outputs", exist_ok=True)
        model.to_file("spa_outputs/pg19_spa.bin")
# model.prune(1)

model.to_file("spa_outputs/pg19_spa.bin")