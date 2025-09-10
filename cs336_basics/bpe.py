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
    tokens = pretokenize(all_text, special_tokens)

    counts = Counter(tokens)

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


def pretokenize(
    all_text: str,
    special_tokens: list[str] | None,
    preserve_special_tokens: bool = False,
) -> list[tuple[bytes, ...]]:
    if not special_tokens:
        splitted_text = [all_text]
    else:
        # sort the special_tokens so that longer tokens are matched first. this can fix the issue of overlapping special tokens. e.g., special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"], the double <|endoftext|><|endoftext|> is preserved as a single token.
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        # Create pattern with capturing groups to preserve special tokens
        pattern = (
            "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
        )
        splitted_text = re.split(pattern, all_text)

    special_tokens_set = set(special_tokens) if special_tokens else set()

    tokens = []
    for text in splitted_text:
        if text not in special_tokens_set:
            tokens.extend(
                [
                    tuple(bytes([b]) for b in word.encode("utf-8"))
                    for word in re.findall(PAT, text)
                ]
            )
        elif preserve_special_tokens:
            tokens.append(
                (text.encode("utf-8"),)
            )  # append bytes of special token as a single value tuple

    return tokens
