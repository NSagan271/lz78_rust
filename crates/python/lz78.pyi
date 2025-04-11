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

    def __init__(self, alphabet_size: int, gamma: float = 0.5,
                compute_training_loss: bool = True,
                store_parent_branches: bool = False,
                max_depth: int = None) -> LZ78SPA:
        pass

    def reset_state(self):
        """
        Reset the state of the LZ78 prefix tree to the root. This can be called,
        e.g., between training on two sequences that should be treated as separate
        sequences.
        """
        pass

    def train_on_block(self, input: Sequence, return_leaf_depths=False, freeze_tree=False) -> dict:
        """
        Use a block of data to update the SPA. If `include_prev_context` is
        true, then this block is considered to be from the same sequence as
        the previous. Otherwise, it is assumed to be a separate sequence, and
        we return to the root of the LZ78 prefix tree.
        
        Inputs:
        - input: input straining sequence
        - return_leaf_depths: whether to also return a list of the depths at
            which the new leaves are added.
        - freeze_tree: whether to only update counts and not add new leaves.
        
        Returns the a dictionary with the self-entropy log loss incurred while
        processing this sequence, as well as possibly the list of depths where
        the new leaves were added.
        """
        pass

    def compute_test_loss(self, input: Sequence, context: Sequence = None,
                          output_per_symbol_losses=False, output_prob_dists=False, output_patch_info=False) -> dict:
        """
        Given the SPA that has been trained thus far, compute the self-entropy
        log loss of a test sequence.

        Returns a dictionary of:
        - average log loss
        - average per-symbol perplexity
        - log loss per symbol (optional)
        - probability distribution per symbol (optional
        - list of "patches" corresponding to indices involved in each path from
            the root to leaf. This is a tuple of (start_idx, end_idx), where the
            end_idx is exclusive. (optional)
        """
        pass

    def compute_test_loss_parallel(self,inputs: list[Sequence], contexts: list[Sequence] = None,
                                   num_threads=16, output_per_symbol_losses=False, output_prob_dists=False,
                                   output_patch_info=False
        ) -> list[dict]:
        """
        Given the SPA that has been trained thus far, compute the self-entropy
        log loss of several test sequences in parallel.

        Returns a list of dictionaries of:
        - average log loss
        - average per-symbol perplexity
        - log loss per symbol (optional)
        - probability distribution per symbol (optional
        - list of "patches" corresponding to indices involved in each path from
            the root to leaf. This is a tuple of (start_idx, end_idx), where the
            end_idx is exclusive. (optional)
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

    def generate_data(self, len: int, seed_data=None, temperature: float=0.5, top_k: int=5) -> tuple[Sequence, float]:
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

    def to_file(self, filename: str):
        pass

    
    def prune(self, min_count: int):
        """
        Prunes the nodes of the tree that have been visited fewer than a
        certain number of times
        """
        pass
 
    
    def get_inference_config(self) -> dict:
        """
        Returns a dictionary of all of the LZ hyperparameters being used for
        inference. See the docstring of self.set_inference_config for
        descriptions of each parameter.
        """
        pass


    def get_generation_config(self) -> dict:
        """
        Returns a dictionary of all of the LZ hyperparameters being used for
        generation. See the docstring of self.set_generation_config for
        descriptions of each parameter.
        """
        pass

    def set_inference_config(self, gamma=None, lb=None, temp=None, lb_or_temp_first=None, adaptive_gamma=None,
        ensemble_type=None, ensemble_n=None, backshift_parsing=None, backshift_ctx_len=None, backshift_break_at_phrase=None):
        """
        Sets the hyperparameters used for inference and SPA computation. Pass
        in a value to change it; otherwise, values will remain at their current
        values by default (see self.get_inference_config for the current
        parameter values).
        
        - gamma: the Dirichlet smoothing hyperparameters for computing the SPA
        
        - lb: a lower bound on the SPA value for any symbol; applied only if
            lb_or_temp_first is not "disabled".
        
        - temp: temperature, applied by modifying the SPA to be
            softmax(2^(spa / temp)); applied only if lb_or_temp_first is not
            "disabled".
        
        - lb_or_temp_first: either "temp_first", "lb_first", or "disabled".
        
        - adaptive_gamma: whether to scale gamma to be smaller for deeper
            nodes, or for nodes that have seen fewer symbols.
            
            Possible Values: either "inverse" (for depth-based adaptive gamma),
            "count", or "disabled".
        
        - ensemble_type: type of ensemble inference to use.
        
            Possible Values: "average" to average the ensemble SPAs, "entropy"
            to weight the average based on the entropy of each SPA, "depth" to
            weight the average based on the node depths, or "disabled".
        
        - ensemble_n: number of nodes in the ensemble; only valid if
            "ensemble_type" is not "disabled".
        
        - backshift_parsing: boolean for whether to enable backshift parsing.
            In backshift parsing, whenever we reach a leaf (or a node that has
            been visited too few times), we return to the root of the tree and
            use the most recently-seen symbols to traverse the tree, hopefully
            arriving at a location with a more accurate SPA.
        
        - backshift_ctx_len: the desired depth to arrive at after backshift
            parsing; i.e., the number of symbols to traverse from the root.
            Only valid if "backshift_parsing" is True.

        - backshift_break_at_phrase: whether to continue backshift parsing
            at a certain shift after a return to the root, or to move on to
            the next shift.
        
        The default value of the parameters (i.e., if you never previously set
        them) is as follows:
            - gamma: 0.5
            - lb: 1e-4
            - temp: 1
            - lb_or_temp_first: lb_first
            - adaptive_gamma: disabled
            - ensemble: disabled
            - backshift_parsing: True
            - backshift_ctx_len: 5
            - backshift_break_at_phrase: False
        """
        pass

    def set_generation_config(self, gamma=None, adaptive_gamma=None, ensemble_type=None, ensemble_n=None,
        backshift_parsing=None, backshift_ctx_len=None, backshift_break_at_phrase=None):
        """
        Set the parameters used for sequence generation. Note that temperature
        and topk are not present here, as they are arguments to the generation
        function itself. See self.get_generation_config for the current
        parameter values.
        
        See self.set_inference_config for descriptions of all parameters and
        their possible values.
        
        The default value of the parameters (i.e., if you never previously set
        them) is as follows:
            - gamma: 0.5
            - adaptive_gamma: disabled
            - ensemble: disabled
            - backshift_parsing: True
            - backshift_ctx_len: 5
            - backshift_break_at_phrase: False
        """
        pass

    def shrink_to_fit(self):
        pass

    def get_total_nodes(self) -> int:
        pass

    def get_total_counts(self) -> int:
        pass


    def get_node_phrase(self, node_idx: int) -> Sequence:
        """
        Get the phrase associated with a specific node of an LZ78 tree,
        as a Sequence object
        """
        pass

    def get_all_node_phrases(self) -> list[Sequence]:
        """
        Get a list of phrases corresponding to all nodes of the LZ78 tree.

        Warning: the return value of this function might be quite large.
        If you don't need all phrases at once and are in a memory-constrained
        environment, it is recommended to loop through get_node_phrase (which
        has some computational overhead, but does not require much memory)
        """
        pass

    def get_all_leaf_ids(self) -> list[int]:
        """
        Gets the node IDs of all leaves
        """
        pass

    def get_spa_at_node_id(self, id: int, gamma=1e-10) -> list[float]:
        """
        Gets the SPA value corresponding to a specific node of the LZ,
        with a specified dirichlet parameter gamma
        """
        pass

    def get_count_at_id(self, id: int) -> int:
        pass


def spa_from_bytes(bytes: bytes) -> LZ78SPA:
    """
    Constructs a trained SPA from its byte array representation.
    """
    pass

def spa_from_file(filename: str) -> LZ78SPA:
    """
    Constructs a trained SPA from a file.
    """
    pass
  
class DirichletLZ78Source:
    def __init__(self, alphabet_size: int, gamma: float, seed=271):
        pass

    def generate_symbols(self, n: int) -> list[int]:
        pass

    def get_log_loss(self) -> float:
        pass

    def get_n(self) -> int:
        pass

    def get_scaled_log_loss(self) -> float:
        pass

class DiracDirichletLZ78Source:
    def __init__(self, gamma, dirichlet_weight, dirac_loc, seed = 271):
        pass

    def generate_symbols(self, n: int) -> list[int]:
        pass

    def get_log_loss(self) -> float:
        pass

    def get_n(self) -> int:
        pass

    def get_scaled_log_loss(self) -> float:
        pass

class DiscreteThetaLZ78Source:
    def __init__(self, theta_values:list[float], probabilities: list[float], seed = 271):
        pass

    def generate_symbols(self, n: int) -> list[int]:
        pass

    def get_log_loss(self) -> float:
        pass

    def get_n(self) -> int:
        pass

    def get_scaled_log_loss(self) -> float:
        pass

def mu_k(seq: list[int], alpha_size: int, k: int) -> float:
    pass

class LZ78Classifier:
    def __init__(self):
        """
        Initializes a classifier with no SPAs.

        SPAs must be added via the add_spa method. Alternatively, see the
        (external to this class) classifier_from_files function.
        """
        pass

    def add_spa(self, spa: LZ78SPA):
        """
        Adds a SPA (corresponding to a class) to the classifier.
        This must be called in order of labels; i.e., the SPA corresponding
        to label 0 must be added first, then label 1, etc.
        """
        pass

    def set_inference_config(self, gamma=None, lb=None, temp=None, lb_or_temp_first=None, adaptive_gamma=None,
        ensemble_type=None, ensemble_n=None, backshift_parsing=None, backshift_ctx_len=None, backshift_break_at_phrase=None):
        """
        Sets the inference config for each SPA. See LZ78SPA.set_inference_config
        for details.
        """
        pass

    def classify(self, input: Sequence) -> int:
        """
        Classifies a single sequence, parallelized across classes.
        """
        pass

    def classify_batch(self, inputs: list[Sequence], num_threads: int) -> list[int]:
        """
        Classifies a batch of sequences, parallelized across input sequences
        but not classes.
        """
        pass

def classifier_from_files(filenames: list[str]) -> LZ78Classifier:
    """
    Loads an LZ78Classifier object from a list of SPAs that have
    been stored to files. The list of filenames must be in order of classes.
    """
    pass

class NGramSPA:
    def __init__(self, alphabet_size: int, n: int, gamma: float = 0.5, ensemble_size: int = 1):
        pass

    def reset_state(self):
        """
        Reset the ngram SPA context
        """
        pass
    
    def train_on_block(self, input: Sequence):
        pass

    def compute_test_loss(
        self, input: Sequence,
        context: Sequence = None,
        output_per_symbol_losses: bool = False,
        output_prob_dists: bool = False
    ) -> dict:
        pass

    def get_counts_for_context(self, context: Sequence) -> list[int]:
        pass
