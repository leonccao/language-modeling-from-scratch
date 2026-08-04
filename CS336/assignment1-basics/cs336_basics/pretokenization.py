import itertools
import os
from collections.abc import Iterator
from typing import BinaryIO

import regex as re

DEBUG = True

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PROCESSES_NUM = 4


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenization(
    text: str,
) -> Iterator[re.Match[str]]:
    # TODO: split on special token, also
    return re.finditer(PATTERN, text)


def find_top_occurence(frequency: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    pair_max, _ = max(frequency.items(), key=lambda item: (item[1], item[0]))
    return pair_max


def merge_pair(
    tokens: tuple[bytes, ...],
    pair_max: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    token_mergerd = pair_max[0] + pair_max[1]
    result: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair_max:
            result.append(token_mergerd)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return tuple(result)


def merge(
    pretokens: dict[tuple[bytes, ...], int], vocab_size: int, vocab: dict[int, bytes]
) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        frequency: dict[tuple[bytes, bytes], int] = {}
        for tokens, cnt in pretokens.items():
            for token, next_token in itertools.pairwise(tokens):
                pair = (token, next_token)
                frequency[pair] = frequency.get(pair, 0) + cnt

        # find top occurency
        pair_max = find_top_occurence(frequency)
        merges.append(pair_max)
        # vocab part 3: merges
        vocab[len(vocab)] = pair_max[0] + pair_max[1]

        new_pretokens: dict[tuple[bytes, ...], int] = {}
        for tokens, cnt in pretokens.items():
            new_tokens = merge_pair(tokens, pair_max)
            new_pretokens[new_tokens] = cnt
        pretokens = new_pretokens

    return merges


"""
TODO
1. draft
2. multiprocessing
3. split on special token
4. lock?
"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    # TODO  multiple tokens
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "rb") as f:
        num_processes = PROCESSES_NUM
        # TODO split tokens
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        pretokens: dict[tuple[bytes, ...], int] = {}
        for start, end in itertools.pairwise(boundaries):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            matches = pretokenization(chunk)
            for match in matches:
                pretoken_str = match.group()
                encoded = pretoken_str.encode("utf-8")
                byte_tokens = tuple(bytes([byte]) for byte in encoded)
                pretokens[byte_tokens] = pretokens.get(byte_tokens, 0) + 1

        # if DEBUG:
        #    print(pretokens)

        # vocab part 1: ascii
        vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
        # vocab part 2: special tokens
        for spec_token in special_tokens:
            vocab[len(vocab)] = spec_token.encode("utf-8")
        merges: list[tuple[bytes, bytes]] = merge(pretokens, vocab_size, vocab)

        if DEBUG:
            print(vocab)
            print(merges)

        return (vocab, merges)
