from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

from cs336_basics.pretokenization import pretoken_match


class Tokenizer:
    id_to_token: dict[int, bytes]
    token_to_id: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_token = vocab
        self.token_to_id: dict[bytes, int] = {token: id for id, token in vocab.items()}
        self.merges = merges

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
        with open(vocab_filepath, encoding="utf-8") as file:
            vocab_data = file.read()
        with open(merges_filepath, encoding="utf-8") as file:
            merges_data = file.read()
        return cls(vocab_data, merges_data, special_tokens)

    def encode_pretoken(self, pretoken: str) -> list[int]:
        result: list[int] = []
        tokens = list(pretoken.encode("utf-8"))
        merged_token: bytes = b""
        for token_num in tokens:
            token = bytes([token_num])
            candidate_token = merged_token + token
            if self.token_to_id.get(candidate_token) is not None:
                merged_token = candidate_token
            else:
                result.append(self.token_to_id.get(merged_token))
                merged_token = token
        result.append(self.token_to_id.get(merged_token))
        return result

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        if self.special_tokens is not None:
            splits: Iterator[str] = re.splititer(
                "|".join(re.escape(token) for token in self.special_tokens), text
            )
        else:
            splits = [text]
        for split in splits:
            matches = pretoken_match(split)
            for match in matches:
                result.extend(self.encode_pretoken(match.group()))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.id_to_token.get(id) for id in ids
        ]
        byte_string = b"".join(tokens)
        return byte_string.decode("utf-8", errors="replace")
