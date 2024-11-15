from typing import Union

class Sequence:
    """
    A Sequence is a list of strings or integers that can be encoded by LZ78.
    Each sequence is associated with an alphabet size, A.

    If the sequence consists of integers, they must be in the range
    {0, 1, ..., A-1}. If A < 256, the sequence is stored internally as bytes.
    Otherwise, it is stored as `uint32`.
    
    If the sequence is a string, a `CharacterMap` object maps each
    character to a number between 0 and A-1.
    
    Inputs:
    - data: either a list of integers or a string.
    - alphabet_size (optional): the size of the alphabet. If this is `None`,
        then the alphabet size is inferred from the data.
    - charmap (optional): A `CharacterMap` object; only valid if `data` is a
        string. If `data` is a string and this is `None`, then the character
        map is inferred from the data.
    """

    def __init__(self, data: Union[list[int], str], alphabet_size: int = None, charmap: CharacterMap = None) -> Sequence:
        pass

    def extend(self, data: Union[list[int], str]) -> None:
        """
        Extend the sequence with new data, which must have the same alphabet
        as the current sequence. If this sequence is represented by a string,
        then `data` will be encoded using the same character map as the
        current sequence.
        """
        pass

    def alphabet_size(self) -> int:
        """
        Returns the alphabet size of the Sequence
        """
        pass

    def get_data(self) -> Union[list[int], str]:
        """
        Fetches the raw data (as a list of integers, or a string) underlying
        this sequence
        """
        pass

    def get_character_map(self) -> CharacterMap:
        """
        If this sequence is represented by a string, returns the underlying
        object that maps characters to integers. Otherwise, this will error.
        """
        pass

class CharacterMap:
    """
    Maps characters in a string to uint32 values in a contiguous range, so that
    a string can be used as an individual sequence. Has the capability to
    **encode** a string into the corresponding integer representation, and
    **decode** a list of integers into a string.

    Inputs:
    - data: a string consisting of all of the characters that will appear in
        the character map. For instance, a common use case is:
        ```
        charmap = CharacterMap("abcdefghijklmnopqrstuvwxyz")
    ```
    """
    def __init__(self, data: str) -> CharacterMap:
        pass

    def encode(self, data: str) -> list[int]:
        """
        Given a string, returns its encoding as a list of integers
        """
        pass

    def decode(self, syms: list[int]) -> str:
        """
        Given a list of integers between 0 and self.alphabet_size() - 1, return
        the corresponding string representation
        """
        pass

    def filter_string(self, data: str) -> str:
        """
        Given a string, filter out all characters that aren't part of the
        mapping and return the resulting string
        """
        pass

    def alphabet_size(self) -> int:
        """
        Returns the number of characters that can be represented by this mapping
        """
        pass

    def add( self, c: str):
        """
        Add a new character to the mapping, if it is not already there.
        """
        pass

    def filter_string_and_replace(self, data: str) -> str:
        """
        Given a string, filter out all characters that aren't part of the
        mapping and replace them with the specified char (which must be
        a part of the character map)
        """
        pass

class CompressedSequence:
    """
    Stores an encoded bitstream, as well as some auxiliary information needed
    for decoding. `CompressedSequence` objects cannot be instantiated directly,
    but rather are returned by `LZ78Encoder.encode`.

    The main functionality is:
    1. Getting the compression ratio as (encoded size) / (uncompressed len * log A),
        where A is the size of the alphabet.
    2. Getting a byte array representing this object, so that the compressed
        sequence can be stored to a file
    """
    def compression_ratio(self) -> float:
        """
        Returns the compression ratio:  (encoded size) / (uncompressed len * log A),
        where A is the size of the alphabet.
        """
        pass

    def to_bytes(self) -> bytes:
        """
        Returns a byte array representing the compressed sequence.

        Common use case: saving to a file,
        ```
        bytearray = compressed_seq.to_bytes()
        with open(filename, 'wb') as f:
            f.write(bytearray)
        ```
        """
        pass

def encoded_sequence_from_bytes(bytes: bytes) -> CompressedSequence:
    """
    Takes a byte array produced by `CompressedSequence.to_bytes` and returns
    the corresponding `CompressedSequence` object 
    """
    pass

class LZ78Encoder:
    """
    Encodes and decodes sequences using LZ78 compression
    """
    def __init__(self) -> LZ78Encoder:
        pass

    def encode(self, input: Sequence) -> CompressedSequence:
        """
        Encodes a `Sequence` object using LZ78 and returns the resulting
        `CompressedSequence`. See "Compression of individual sequences via
        variable-rate coding" (Ziv, Lempel 1978) for more details. 
        """
        pass

    def decode(self, input: CompressedSequence) -> Sequence:
        """
        Decodes a sequence compressed via `LZ78Encoder.encode`
        """
        pass

class BlockLZ78Encoder:
    """
    Block LZ78 encoder: you can pass in the input sequence to be compressed
    in chunks, and the output (`encoder.get_encoded_sequence()`) is as if the
    full concatenated sequence was passed in to an LZ78 encoder
    """

    def __init__(self, alpha_size: int) -> BlockLZ78Encoder:
        pass

    def encode_block(self, input: Sequence) -> None:
        """
        Encodes a block using LZ78, starting at the end of the previous block.
    
        All blocks passed in must be over the same alphabet. For character
        sequences, they must use the same `CharacterMap` (i.e., the same chars
        are mapped to the same symbols; they need not use the exact same
        `CharacterMap` instance).
        
        The expected alphabet is defined by the first call to `encode_block`,
        and subsequent calls will error if the input sequence has a different
        alphabet size or character map.
        """
        pass

    def alphabet_size(self) -> int:
        """
        Returns the alphabet size passed in upon instantiation
        """
        pass

    def get_encoded_sequence(self) -> CompressedSequence:
        """
        Returns the `CompressedSequence` object, which is equivalent to the
        output of `LZ78Encoder.encode` on the concatenation of all inputs to
        `encode_block` thus far.
        
        Errors if no blocks have been compressed so far.
        
        """
        pass

    def decode(self) -> Sequence:
        """
        Performs LZ78 decoding on the compressed sequence that has been
        generated thus far.
        
        Errors if no blocks have been compressed so far.
        """
        pass

class LZ78SPA:
    """
    Constructs a sequential probability assignment on input data via LZ78
    incremental parsing. This is the implementation of the family of SPAs
    described in "A Family of LZ78-based Universal Sequential Probability
    Assignments" (Sagan and Weissman, 2024), under a Dirichlet(gamma) prior.

    Under this prior, the sequential probability assignment is an additive
    perturbation of the emprical distribution, conditioned on the LZ78 prefix
    of each symbol (i.e., the probability model is proportional to the
    number of times each node of the LZ78 tree has been visited, plus gamma).

    This SPA has the following capabilities:
    - training on one or more sequences,
    - log loss ("perplexity") computation for test sequences,
    - SPA computation (using the LZ78 context reached at the end of parsing
        the last training block),
    - sequence generation.

    Note that the LZ78SPA does not perform compression; you would have to use
    a separate BlockLZ78Encoder object to perform block-wise compression.
    """

    def __init__(self, alphabet_size: int, gamma: float = 0.5, debug: bool = False) -> LZ78SPA:
        pass

    def reset_state(self):
        """
        Reset the state of the LZ78 prefix tree to the root. This can be called,
        e.g., between training on two sequences that should be treated as separate
        sequences.
        """
        pass

    def train_on_block(self, input: Sequence) -> float:
        """
        Use a block of data to update the SPA. If `include_prev_context` is
        true, then this block is considered to be from the same sequence as
        the previous. Otherwise, it is assumed to be a separate sequence, and
        we return to the root of the LZ78 prefix tree.
        
        Returns the self-entropy log loss incurred while processing this
        sequence.
        """
        pass

    def compute_test_loss(self, input: Sequence) -> float:
        """
        Given the SPA that has been trained thus far, compute the self-entropy
        log loss ("perplexity") of a test sequence. `include_prev_context` has
        the same meaning as in `train_on_block`.
        """
        pass

    def compute_spa_at_current_state(self) -> list[float]:
        """
        Computes the SPA for every symbol in the alphabet, using the LZ78
        context reached at the end of parsing the last training block
        """
        pass

    def get_normalized_log_loss(self) -> float:
        """
        Returns the normaliized self-entropy log loss incurred from training
        the SPA thus far.
        """
        pass

    def generate_data(self, len: int, seed_data=None, temperature: float=0.5, top_k: int=5,
                      desired_context_length: int=10, min_spa_training_points: int=1) -> tuple[Sequence, float]:
        """
        Generates a sequence of data, using temperature and top-k sampling (see
        the "Experiments" section of [Sagan and Weissman 2024] for more details).
        
        Inputs:
        - len: number of symbols to generate
        - seed_data: you can specify that the sequence of generated data
            be the continuation of the specified sequence.
        - temperature: a measure of how "random" the generated sequence is. A
            temperature of 0 deterministically generates the most likely
            symbols, and a temperature of 1 samples directly from the SPA.
            Temperature values around 0.1 or 0.2 function well.
        - top_k: forces the generated symbols to be of the top_k most likely
            symbols at each timestep.
        - desired_context_length: the SPA tries to maintain a context of at least a
            certain length at all times. So, when we reach a leaf of the LZ78
            prefix tree, we try traversing the tree with different suffixes of
            the generated sequence until we get a sufficiently long context
            for the next symbol.
        - min_spa_training_points: requires that a node of the LZ78 prefix tree
            has been visited at least this number of times during training before
            it can be used for generation. i.e., instead of returning to the
            root upon reaching a leaf, we would return to the root once we reach
            any node that has not been traversed enough times.
        
        Returns a tuple of the generated sequence and that sequence's log loss,
        or perplexity.
        
        Errors if the SPA has not been trained so far, or if the seed data is
        not over the same alphabet as the training data.
        """
        pass

    def to_bytes(self) -> bytes:
        """
        Returns a byte array representing the trained SPA, e.g., to save the
        SPA to a file.
        """
        pass

    def get_debug_info(self) -> LZ78DebugInfo:
        """
        Extracts information about the depth of leaves of the LZ78 prefix tree
        underlying this SPA.
        """
        pass

def spa_from_bytes(bytes: bytes) -> LZ78SPA:
    """
    Constructs a trained SPA from its byte array representation.
    """
    pass

class LZ78DebugInfo:
    """
    Debugging information for the LZ78SPA; i.e., the depths of the leaves.
    """

    def get_max_leaf_depth(self) -> int:
        """
        Get the length of the longest branch of the LZ78 prefix tree
        """
        pass

    def get_min_leaf_depth(self) -> int:
        """
        Get the length of the shortest branch of the LZ78 prefix tree
        """
        pass

    def get_mean_leaf_depth(self) -> float:
        """
        Get the (unweighted) average depth of all leaves of the LZ78 prefix tree
        """
        pass

    def get_leaf_depths(self) -> list[int]:
        """
        Returns the depth of all leaves of the LZ78 tree as a list (in no
        particular order)
        """
        pass

    def get_longest_branch(self) -> list[int]:
        """
        Returns the longest LZ78 phrase encoded by the prefix tree,
        as a list of integer symbols
        """
        pass