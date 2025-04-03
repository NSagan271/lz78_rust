import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Union

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


class EmbeddingModel:
    def __init__(self, model_str: str, tokenizer_str: str = None, max_length=8192, device="cpu"):
        if tokenizer_str is None:
            tokenizer_str = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str, trust_remote_code=True, device=device)
        self.model = AutoModel.from_pretrained(model_str, trust_remote_code=True, device_map=device)
        self.max_length = max_length
        self.device = device

    def embed(self, text: Union[str, list[str]], normalize=True):
        if isinstance(text, str):
            text = [text]
        batch_dict = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model(**batch_dict)

        embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        if normalize:
            return F.normalize(embeds, p=2, dim=1)
        return embeds