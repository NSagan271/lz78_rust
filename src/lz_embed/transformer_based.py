import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Union
from lz_embed.utils import AlphabetInfo, LZEmbedding
from lz78 import LZ78SPA


def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """
    From documentation of https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class DeepEmbeddingModel:
    def __init__(self, model_str: str, tokenizer_str: str = None, max_length=8192, device="cpu"):
        if tokenizer_str is None:
            tokenizer_str = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str, trust_remote_code=True, device=device)
        self.model = AutoModel.from_pretrained(model_str, trust_remote_code=True, device_map=device)
        self.max_length = max_length
        self.device = device

    def to(self, device: str):
        self.device = device
        self.model = self.model.to(device)
        self.tokenizer = self.tokenizer.to(device)

    def embed(self, text: Union[str, list[str]], normalize=True):
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            batch_dict = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**batch_dict)

            embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            if normalize:
                return F.normalize(embeds, p=2, dim=1)
            return embeds
        

class LZPlusEmbeddingModel(LZEmbedding):
    def __init__(
            self, model: DeepEmbeddingModel, alpha_info: AlphabetInfo,
            max_depth: int = None, backshift_parsing_len: int = None):
        super().__init__(alpha_info, max_depth)
        self.model = model
        self.trained = False
        self.spa = LZ78SPA(alphabet_size=self.alphabet_size,
                           compute_training_loss=False,
                           max_depth=max_depth)
        if backshift_parsing_len:
            self.spa.set_inference_config(
                backshift_parsing=True,
                backshift_ctx_len=backshift_parsing_len
            )

        # compute fixed length
        self.length = self.model.embed("hello").shape[1]

    def train(self, sequence: Union[str, list[int]]):
        sequence = self.validate_seq(sequence)
        self.spa.reset_state()
        self.spa.train_on_block(sequence)
        self.trained = True

    def fixed_length(self):
        return self.length

    def embed_single(self, sequence: Union[str, list[int]])-> torch.Tensor:
        assert self.trained, "Must train LZPlusEmbeddingModel with model.train(sequence) first!"
        sequence = self.validate_seq(sequence)
        res = self.spa.compute_test_loss(sequence, output_patch_info=True, output_per_symbol_losses=True)
        log_losses = torch.Tensor(res["log_losses"])
        
        patches = res["patch_info"]
        weights = torch.Tensor([(2**(-log_losses[start:end].mean())) for (start, end) in patches]).to(self.model.device)
        weights /= weights.sum()

        sequence = sequence.get_data()
        texts = [sequence[start:end] for (start, end) in patches]
        embeds = self.model.embed(texts, normalize=True)
        return (embeds * weights.unsqueeze(1)).sum(dim=0)
