import ast
import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

from cs336_basics.pretokenization import pretoken_match

DEBUG = False


class Tokenizer:
    id_to_token: dict[int, bytes]
    token_to_id: dict[bytes, int]
    merges: dict[tuple[bytes, bytes], int]
    special_tokens: list[str]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_token = vocab
        self.token_to_id = {token: id for id, token in vocab.items()}
        self.merges = {}
        for merge in merges:
            merged_token = merge[0] + merge[1]
            self.merges[merge] = self.token_to_id.get(merged_token)

        self.special_tokens = special_tokens
        if special_tokens is not None:
            for special_token in special_tokens:
                encoded_token = special_token.encode("utf-8")
                if self.token_to_id.get(encoded_token) is None:
                    id = len(self.id_to_token)
                    self.id_to_token[id] = encoded_token
                    self.token_to_id[encoded_token] = id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: Path,
        merges_filepath: Path,
        special_tokens: list[str] | None = None,
    ):
        """Load one byte-exact Python tuple per line from tokenizer artifacts."""
        with open(vocab_filepath, encoding="utf-8") as file:
            serialized_vocab = [
                ast.literal_eval(line) for line in file if line.strip()
            ]
        with open(merges_filepath, encoding="utf-8") as file:
            merges = [ast.literal_eval(line) for line in file if line.strip()]

        if not all(
            isinstance(token_id, int) and isinstance(token, bytes)
            for token_id, token in serialized_vocab
        ):
            raise ValueError("Vocabulary entries must be (int, bytes) pairs.")
        if not all(
            isinstance(left_token, bytes) and isinstance(right_token, bytes)
            for left_token, right_token in merges
        ):
            raise ValueError("Merge entries must be (bytes, bytes) pairs.")

        vocab = dict(serialized_vocab)
        return cls(vocab, merges, special_tokens)

    """
    TODO
    1. brute-force
    2. linkedlist
    3. heap
    """

    def encode_pretoken(self, pretoken: str) -> list[int]:
        tokens_code = list(pretoken.encode("utf-8"))
        tokens = [bytes([token]) for token in tokens_code]

        while True:
            min_rank: int | None = None
            min_merge: tuple[bytes, bytes] | None = None
            for token, next_token in itertools.pairwise(tokens):
                rank = self.merges.get((token, next_token))
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_rank = rank
                    min_merge = (token, next_token)

            if min_rank is None or min_merge is None:
                break

            i = 0
            new_tokens: list[bytes] = []
            while i < len(tokens):
                token = tokens[i]

                if i < len(tokens) - 1:
                    next_token = tokens[i + 1]
                    if token == min_merge[0] and next_token == min_merge[1]:
                        new_tokens.append(token + next_token)
                        i += 2
                        continue

                new_tokens.append(token)
                i += 1

            tokens = new_tokens

        result: list[int] = [self.token_to_id.get(token) for token in tokens]
        return result

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        special_tokens = sorted(self.special_tokens or [], key=len, reverse=True)
        if special_tokens:
            pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
            splits: Iterator[str] = re.splititer(pattern, text)
        else:
            splits = iter([text])

        special_tokens_set = set(special_tokens)

        for split in splits:
            if DEBUG:
                print("split")
                print(split)
            if split in special_tokens_set:
                result.append(self.token_to_id.get(split.encode("utf-8")))
                continue
            for match in pretoken_match(split):
                result.extend(self.encode_pretoken(match.group()))

        if DEBUG:
            print("encode result")
            print(result)
            print(self.decode(result))

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens: list[bytes] = [self.id_to_token.get(id) for id in ids]
        byte_string = b"".join(tokens)
        return byte_string.decode("utf-8", errors="replace")
