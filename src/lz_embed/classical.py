
from lz78 import LZ78SPA, CharacterMap, Sequence
from typing import Union
import torch


def seq_to_int(seq: Sequence, alphabet_size: int):
    if len(seq) == 0:
        return 0
    val = 1
    for i in range(len(seq)):
        val += (alphabet_size**i) * seq[i]
    return val


def basic_lz_spectrum(sequence: Union[str, list[int]], valid_characters: str = None,
                      alphabet_size: int = None, max_depth = None) -> list[float]:
    if isinstance(sequence, str):
        assert valid_characters is not None, "For a string sequence, must specify valid_characters argument"
        charmap = CharacterMap(sequence)
        sequence = Sequence(sequence, charmap=charmap)
        alphabet_size = charmap.alphabet_size()
    else:
        assert alphabet_size is not None, "For an integer sequence, must specify alphabet_size argument"
        sequence = Sequence(sequence, alphabet_size=alphabet_size)

    spa = LZ78SPA(alphabet_size=alphabet_size, compute_training_loss=False, max_depth=max_depth)
    spa.train_on_block(sequence)

    all_node_phrases = spa.get_all_node_phrases()
    spectrum_idx_to_node_id = {}
    max_spectrum_idx = 0
    for (i, phrase) in enumerate(all_node_phrases):
        if spa.get_count_at_id(i) == 0:
            continue
        x = seq_to_int(phrase, alphabet_size)
        max_spectrum_idx = max(max_spectrum_idx, x)
        spectrum_idx_to_node_id[x] = i

    res = torch.ones((max_spectrum_idx + 1) * (alphabet_size - 1)) * (1/alphabet_size)
    for idx in spectrum_idx_to_node_id:
        print(idx, spectrum_idx_to_node_id[idx], spa.get_spa_at_node_id(spectrum_idx_to_node_id[idx], gamma=0))
        start = idx * (alphabet_size - 1)
        end = (idx + 1) * (alphabet_size - 1)
        res[start:end] = torch.Tensor(spa.get_spa_at_node_id(spectrum_idx_to_node_id[idx], gamma=0)[:-1])
    
    return res


    
    
