from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
import torch
from lz78 import CharacterMap, Sequence

@dataclass
class AlphabetInfo:
    alphabet_size: int = None
    valid_character_string: str = None


class LZEmbedding(ABC):
    def __init__(self, alpha_info: AlphabetInfo, max_depth: int = None):
        if alpha_info.valid_character_string is not None:
            self.charmap = CharacterMap(alpha_info.valid_character_string)
            self.alphabet_size = self.charmap.alphabet_size()
        else:
            assert alpha_info.alphabet_size is not None, "AlphabetInfo cannot be empty"
            self.alphabet_size = alpha_info.alphabet_size
            self.charmap = None
        self.max_depth = max_depth

    def validate_seq(self, sequence:  Union[str, list[int]]) -> Sequence:
        if isinstance(sequence, str):
            assert self.charmap is not None, "Expected an integer sequence, got a string"
            sequence = self.charmap.filter_string(sequence)
            return Sequence(sequence, charmap=self.charmap)
        else:
            assert self.charmap is None, "Expected a string, got an integer sequence"
            return Sequence(sequence, alphabet_size=self.alphabet_size)

    @abstractmethod
    def fixed_length(self):
        return None
    
    @abstractmethod
    def embed_single(self, sequence: Union[str, list[int]]) -> torch.Tensor:
        raise NotImplementedError()

    def embed(self, sequences: list[Union[str, list[int]]],) -> Union[torch.Tensor, list[torch.Tensor]]:
        fixed_len = self.fixed_length()
        if fixed_len is not None:
            embeds = torch.ones((len(sequences), fixed_len)) / self.alphabet_size
            for i in range(len(sequences)):
                embeds[i, :] = self.embed_single(sequences[i])
            return embeds
        return [self.embed_single(seq) for seq in sequences]