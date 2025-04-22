
from lz78 import LZ78SPA, Sequence, NGramSPA
from typing import Union
import torch
from torch import Tensor
from lz_embed.utils import AlphabetInfo, LZEmbedding
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card import SentenceTransformerModelCardData
from transformers.configuration_utils import PretrainedConfig
import os


def seq_to_int(seq: Sequence, alphabet_size: int):
    val = 0
    for i in range(len(seq)):
        val += (alphabet_size ** i)
    for i in range(len(seq)):
        val += (alphabet_size**i) * (seq[i])
    return val

    
class BasicLZSpectrum(LZEmbedding):
    def __init__(self, alpha_info: AlphabetInfo,
                 max_depth: int = None, fixed_len: int = None,
                 lowercase: bool = True):
        super().__init__(alpha_info, max_depth)
        if fixed_len is None and max_depth is not None:
            fixed_len = 1
            for i in range(1, max_depth+1):
                fixed_len += self.alphabet_size**i
            fixed_len *= (self.alphabet_size - 1)
        self.fixed_len = fixed_len

        self.lowercase = lowercase

    def fixed_length(self):
        return self.fixed_len
    
    def encode_single(self, sequence: Union[str, list[int]]) -> torch.Tensor:
        fixed_len = self.fixed_len
        if type(sequence) == str and self.lowercase:
            sequence = sequence.lower()
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
        
        return res[:fixed_len] - 1/self.alphabet_size


class BasicNGramSpectrum(LZEmbedding):
    def __init__(self, alpha_info: AlphabetInfo,
                 n: int = 4, lowercase: bool = True,
                 normalized_subspace: bool = False):
        super().__init__(alpha_info, n)
        self.n = n
        self.fixed_len = (self.alphabet_size - 1) * self.alphabet_size**(n)

        self.lowercase = lowercase
        self.normalized_subspace = normalized_subspace

    def fixed_length(self):
        return self.fixed_len

    def encode_single(self, sequence: str | list[int]) -> torch.Tensor:
        if type(sequence) == str and self.lowercase:
            sequence = sequence.lower()
        sequence = self.validate_seq(sequence)
        spa = NGramSPA(self.alphabet_size, self.n)

        spa.train_on_block(sequence)
        return torch.Tensor(spa.to_vec(self.normalized_subspace)) - (1 / self.alphabet_size)
    
    def encode(self, sequences) -> Union[torch.Tensor, list[torch.Tensor]]:
        if type(sequences) == list[int]:
            return self.encode_single(sequences).cpu()
        
        embeds = torch.ones((len(sequences), self.fixed_len)) / self.alphabet_size
        for i in range(len(sequences)):
            embeds[i, :] = self.encode_single(sequences[i]).cpu()                    
        return embeds


class NGramSpectrumEmbeddingConfig(PretrainedConfig):
    def __init__(
        self,
        alphabet_info: AlphabetInfo = None,
        n: int = None,
        lowercase: bool = True,
        normalize: bool = True,
        use_pca: bool = False,
        pca_dim: int = 1024,
        model_name: str = "lz/ngram_spectrum",
        **kwargs
    ):
        if alphabet_info:
            self.valid_character_string = alphabet_info.valid_character_string
            self.alphabet_size = alphabet_info.alphabet_size
            
        self.n = n
        self.lowercase = lowercase
        self.normalize = normalize
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.model_name = model_name

        super().__init__(**kwargs)
        

class NGramSpectrumEmbedding(SentenceTransformer):
    config_class = NGramSpectrumEmbeddingConfig
    base_model_prefix = "ngram"
    supports_gradient_checkpointing = False
    _no_split_modules = []
    
    def __init__(
        self, config:NGramSpectrumEmbeddingConfig,
        model_save_path: str
    ):
        super().__init__()

        self.config = config
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)

        # model to compute the ngram
        self.ngram_spectrum = BasicNGramSpectrum(
            AlphabetInfo(config.alphabet_size,config.valid_character_string),
            config.n, config.lowercase,
            normalized_subspace=False
        )

        # needed for SentenceTransformer
        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name=config.model_name
        )

        # PCA subspace
        self.subspace = None
        self.use_pca = config.use_pca
        self.pca_num_docs_used = 0
        if config.use_pca:
            subspace_path = os.path.join(model_save_path, f"subspace_{config.pca_dim}.pkl")
            if os.path.exists(subspace_path):
                self.subspace = torch.load(subspace_path)
                with open(os.path.join(model_save_path, f"pca_num_docs_used.txt")) as f:
                    self.pca_num_docs_used = int(f.read())
            else:
                print("Warning: use_pca specified True, but no saved PCA subspaces found.")
                print("Please train the PCA subspace using model.train_subspace(dataloader)")

        self.normalize = config.normalize

    def save_pretrained(self):
        self.config.save_pretrained(self.model_save_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = NGramSpectrumEmbeddingConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
        # Initialize the model
        model = cls(config, pretrained_model_name_or_path)
        return model

    def train_subspace(self, dataloader=None, device="cuda"):
        mtx_path = f"{self.model_save_path}/mtx.pkl"

        if os.path.exists(mtx_path):
            mtx = torch.load(mtx_path, map_location=device)
        else:
            assert dataloader, f"If a dataloader is not specified for train_subspace, then the data matrix must exist at {mtx_path}"
            mtx = torch.zeros((0, self.ngram_spectrum.fixed_len), device=device)

        if dataloader:   
            for (i, document) in enumerate(tqdm(dataloader)):
                self.pca_num_docs_used += 1
                mtx = torch.concat((mtx, self.ngram_spectrum.encode([document]).to(mtx.dtype).to(device)), axis=0)
            print(f"Saving data matrix to {mtx_path}")
            torch.save(mtx, f=mtx_path)
            with open(os.path.join(self.model_save_path, f"pca_num_docs_used.txt"), "w") as f:
                f.write(str(self.pca_num_docs_used))

        print(f"Computing subspace")
        mtx -= mtx.mean(dim=0) # subtract the mean of each feature
        _, _, subspace = torch.svd_lowrank(mtx, q=self.config.pca_dim*3)
        self.subspace = subspace[:, :self.config.pca_dim].cpu()

        subspace_path =f"{self.model_save_path}/subspace_{self.config.pca_dim}.pkl"
        print(f"Saving subspace to {subspace_path}")
        torch.save(self.subspace, f=subspace_path)
        print("Done saving")

    def tokenize(self, texts):
        return {
            "texts": texts
        }
    
    def forward(self, input: dict[str, list], **kwargs) -> dict[str, Tensor]:
        embeds = self.ngram_spectrum.encode(input["texts"])
        if self.use_pca:
            embeds = embeds @ self.subspace

        if self.normalize:
            embeds /= (embeds.norm(dim=1, keepdim=True) + 1e-32)

        return {
            "sentence_embedding": embeds
        }
    
    
