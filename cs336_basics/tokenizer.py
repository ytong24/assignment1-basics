from typing import Iterable, Iterator
from collections import Counter

from cs336_basics.bpe import pretokenize


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.special_tokens_bytes = (
            set([token.encode("utf-8") for token in special_tokens])
            if special_tokens
            else set()
        )

        # used when encoding
        self.reverse_vocab = {
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        # TODO:
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # pre-tokenize
        tokens = pretokenize(text, self.special_tokens, preserve_special_tokens=True)

        # merge
        def merge_one(token: tuple[bytes, ...]) -> tuple[bytes, ...]:
            token_l = list(token)
            for merge_pair in self.merges:
                # try to merge first and second in token_l
                new_token_l = []
                num_special_token = 0
                i = 0
                while i < len(token_l) - 1:
                    if token_l[i] in self.special_tokens_bytes:
                        new_token_l.append(token_l[i])
                        num_special_token += 1
                        i += 1
                    elif (token_l[i], token_l[i + 1]) == merge_pair:

                        new_token_l.append(token_l[i] + token_l[i + 1])
                        i += 2
                    else:
                        new_token_l.append(token_l[i])
                        i += 1

                if i == len(token_l) - 1:
                    new_token_l.append(token_l[i])

                token_l = new_token_l

                # early break: the number of remaining non-special tokens is less than 2
                if len(token_l) - num_special_token < 2:
                    break

            return tuple(token_l)

        encoding = [self.reverse_vocab[b] for token in tokens for b in merge_one(token)]

        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory
        """
        return (token_id for text in iterable for token_id in self.encode(text))

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return (b"".join([self.vocab[id] for id in ids])).decode(
            "utf-8", errors="replace"
        )
