
from lz78 import LZ78SPA, Sequence, NGramSPA
from typing import Union
import torch
from lz_embed.utils import AlphabetInfo, LZEmbedding
from tqdm import tqdm
import numpy as np
    

def seq_to_int(seq: Sequence, alphabet_size: int):
    val = 0
    for i in range(len(seq)):
        val += (alphabet_size ** i)
    for i in range(len(seq)):
        val += (alphabet_size**i) * (seq[i])
    return val


class BasicLZSpectrum(LZEmbedding):
    def __init__(self, alpha_info: AlphabetInfo, max_depth: int = None, fixed_len: int = None):
        super().__init__(alpha_info, max_depth)
        if fixed_len is None and max_depth is not None:
            fixed_len = 1
            for i in range(1, max_depth+1):
                fixed_len += self.alphabet_size**i
            fixed_len *= (self.alphabet_size - 1)
        self.fixed_len = fixed_len

    def fixed_length(self):
        return self.fixed_len
    
    def encode_single(self, sequence: Union[str, list[int]]) -> torch.Tensor:
        fixed_len = self.fixed_len
        sequence = self.validate_seq(sequence)

        spa = LZ78SPA(alphabet_size=self.alphabet_size, compute_training_loss=False, max_depth=self.max_depth)
        spa.train_on_block(sequence)

        all_node_phrases = spa.get_all_node_phrases()
        spectrum_idx_to_node_id = {}
        max_spectrum_idx = 0
        for (i, phrase) in enumerate(all_node_phrases):
            if spa.get_count_at_id(i) == 0:
                continue
            x = seq_to_int(phrase, self.alphabet_size)

            max_spectrum_idx = max(max_spectrum_idx, x)
            spectrum_idx_to_node_id[x] = i

        vec_len = (max_spectrum_idx + 1) * (self.alphabet_size - 1)
        if fixed_len is not None:
            vec_len = max(vec_len, fixed_len)
        else:
            fixed_len = vec_len

        res = torch.ones(vec_len) * (1/self.alphabet_size)
        for idx in spectrum_idx_to_node_id:
            start = idx * (self.alphabet_size - 1)
            end = (idx + 1) * (self.alphabet_size - 1)
            res[start:end] = torch.Tensor(spa.get_spa_at_node_id(spectrum_idx_to_node_id[idx], gamma=0)[:-1])
        
        return res[:fixed_len]
    
class BasicNGramSpectrum:
    def __init__(self, alpha_size: int, n: int = 4):
        self.alphabet_size = alpha_size
        self.n = n

        fixed_len = 1
        for i in range(1, n+1):
            fixed_len += self.alphabet_size**i
        fixed_len *= (self.alphabet_size - 1)
        self.fixed_len = fixed_len

    def encode_single(self, sequence: list[int]) -> torch.Tensor:
        spa = NGramSPA(self.alphabet_size, self.n, ensemble_size=self.n)
        spa.train_on_block(Sequence(sequence, alphabet_size=self.alphabet_size))
        embed = torch.ones(self.fixed_len) / (self.alphabet_size)
        idx = 0
        for i in range(0, self.n+1):
            ctxs = np.array(np.meshgrid(*([list(range(self.alphabet_size)) for _ in range(i)]))).T.reshape(-1,i) if i > 0 else [[]]
            for ctx in ctxs:
                counts = torch.Tensor(spa.get_counts_for_context(Sequence(ctx, alphabet_size=self.alphabet_size)))
                if counts.sum() == 0:
                    idx += self.alphabet_size - 1
                    continue
                counts /= counts.sum()
                embed[idx:idx+self.alphabet_size-1] = counts[:-1]
                idx += self.alphabet_size - 1
        
        return embed
    
    def encode(self, sequences) -> Union[torch.Tensor, list[torch.Tensor]]:
        if type(sequences) == list[int]:
            return self.encode_single(sequences).cpu()
        
        embeds = torch.ones((len(sequences), self.fixed_len)) / self.alphabet_size
        for i in tqdm(range(len(sequences))):
            embeds[i, :] = self.encode_single(sequences[i]).cpu()                    
        return embeds
    
    
