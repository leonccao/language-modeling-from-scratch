import itertools
import os
from collections.abc import Iterator
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

DEBUG = True

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PROCESSES_NUM = 4


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for token in split_special_tokens:
        assert isinstance(token, bytes), "Must represent special token as a bytestring"

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
            found_at = max(mini_chunk.find(token) for token in split_special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def regex_match(
    text: str,
) -> Iterator[re.Match[str]]:
    return re.finditer(PATTERN, text)


def find_top_freq(frequency: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
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


def record_pairs(
    pretoks: dict[tuple[bytes, ...], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, ...], list[int]]]:
    loc: dict[tuple[bytes, ...], list[int]] = {}
    freq: dict[tuple[bytes, bytes], int] = {}

    for tokens, cnt in pretoks.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            freq[pair] = freq.get(pair, 0) + cnt
            loc.get(pair, []).append(i)

    return (freq, loc)


def merge(
    pretoks: dict[tuple[bytes, ...], int], vocab_size: int, vocab: dict[int, bytes]
) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        freq, _ = record_pairs(pretoks)

        # find most frequent pair
        pair_max = find_top_freq(freq)
        merges.append(pair_max)
        # vocab part 3: merges
        vocab[len(vocab)] = pair_max[0] + pair_max[1]

        new_pretoks: dict[tuple[bytes, ...], int] = {}
        for tokens, cnt in pretoks.items():
            new_tokens = merge_pair(tokens, pair_max)
            new_pretoks[new_tokens] = cnt
        pretoks = new_pretoks

    return merges


def tokenization(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    split_chunks: list[str] = re.split(
        "|".join(re.escape(token) for token in special_tokens), chunk
    )

    pretokens: dict[tuple[bytes, ...], int] = {}
    for split_chunk in split_chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        matches = regex_match(split_chunk)
        for match in matches:
            pretoken_str = match.group()
            encoded = pretoken_str.encode("utf-8")
            byte_tokens = tuple(bytes([byte]) for byte in encoded)
            pretokens[byte_tokens] = pretokens.get(byte_tokens, 0) + 1
    return pretokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "rb") as f:
        num_processes = PROCESSES_NUM
        special_tokens_bytes: list[bytes] = [
            special_token.encode("utf-8") for special_token in special_tokens
        ]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks: list[str] = []
        for start, end in itertools.pairwise(boundaries):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        with Pool(processes=num_processes) as pool:
            partial_tokens = pool.starmap(
                tokenization, zip(chunks, itertools.repeat(special_tokens))
            )

        pretokens: dict[tuple[bytes, ...], int] = {}
        for partial in partial_tokens:
            for tokens, count in partial.items():
                pretokens[tokens] = pretokens.get(tokens, 0) + count

        """
        if DEBUG:
            print(pretokens)
        """

        # vocab part 1: ascii
        vocab: dict[int, bytes] = {
            token_id: bytes([token_id]) for token_id in range(256)
        }
        # vocab part 2: special tokens
        for spec_token in special_tokens:
            vocab[len(vocab)] = spec_token.encode("utf-8")
        merges: list[tuple[bytes, bytes]] = merge(pretokens, vocab_size, vocab)

        """
        if DEBUG:
            print(vocab)
            print(merges)
        """

        return (vocab, merges)
