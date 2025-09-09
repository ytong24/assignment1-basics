from collections import Counter, defaultdict
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # vocabulary initialization
    # 256 byte values + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode()

    # pre-tokenization
    with open(input_path, "r", encoding="utf-8") as f:
        all_text = f.read()
    splitted_text = re.split("|".join(special_tokens), all_text)
    tokens = []
    for text in splitted_text:
        tokens.extend(re.findall(PAT, text))
    counts = {
        tuple(bytes([b]) for b in token.encode()): count
        for token, count in Counter(tokens).items()
    }

    # compute BPE merges
    merges = []
    while len(vocab) < vocab_size:
        # get byte_pair count
        bp_counts = defaultdict(int)
        for bytes_tuple, count in counts.items():
            for first_b, second_b in zip(bytes_tuple[:-1], bytes_tuple[1:]):
                bp_counts[(first_b, second_b)] += count

        # get the most frequent bytes pair
        most_freq_bp = max(bp_counts, key=lambda k: (bp_counts[k], k))
        merges.append(most_freq_bp)
        vocab[len(vocab)] = most_freq_bp[0] + most_freq_bp[1]

        # udpate the counts dict
        new_counts = {}
        for bytes_tuple, count in counts.items():
            l = []
            i = 0
            while i < len(bytes_tuple) - 1:
                if (bytes_tuple[i], bytes_tuple[i + 1]) == merges[-1]:
                    l.append(bytes_tuple[i] + bytes_tuple[i + 1])
                    i += 2
                else:
                    l.append(bytes_tuple[i])
                    i += 1
            if i == len(bytes_tuple) - 1:
                l.append(bytes_tuple[i])

            new_counts[tuple(l)] = count

        counts = new_counts

    return vocab, merges
