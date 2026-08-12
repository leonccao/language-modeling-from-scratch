import argparse
import itertools
import os
from collections.abc import Iterator
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization import find_chunk_boundaries, pretoken_match

DEBUG = False

WORKERS_NUM = 4
CHUNKS_NUM = 128

Pair = tuple[bytes, bytes]


def find_top_freq(
    frequency: dict[Pair, int],
) -> tuple[Pair, int]:
    pair_max, freq_max = max(frequency.items(), key=lambda item: (item[1], item[0]))
    return (pair_max, freq_max)


def dec_pair(
    pretok_id: int,
    cnt: int,
    pair: Pair,
    freq: dict[Pair, int],
    aprs: dict[Pair, dict[int, int]],
):
    if DEBUG:
        print("pretok_id")
        print(pretok_id)
        print("pair")
        print(pair)
        print("aprs")
        print(aprs)
        print("apr_dict")
        idx = aprs.get(pair, 0)
        print(idx)

    new_freq = freq.pop(pair) - cnt
    if new_freq > 0:
        freq[pair] = new_freq

    aprs_dict: dict[int, int] = aprs.pop(pair)
    new_aprs_cnt = aprs_dict.pop(pretok_id) - cnt
    if new_aprs_cnt > 0:
        aprs_dict[pretok_id] = new_aprs_cnt
    if len(aprs_dict) > 0:
        aprs[pair] = aprs_dict


def inc_pair(
    pretok_id: int,
    cnt: int,
    pair: Pair,
    freq: dict[Pair, int],
    aprs: dict[Pair, dict[int, int]],
):
    freq[pair] = freq.get(pair, 0) + cnt

    aprs_dict: dict[int, int] = aprs.get(pair, {})
    aprs_dict[pretok_id] = aprs_dict.get(pretok_id, 0) + cnt
    aprs[pair] = aprs_dict


def merge_pairs(
    pretok_id: int,
    tokens: tuple[bytes, ...],
    cnt: int,
    pair_max: Pair,
    freq: dict[Pair, int],
    aprs: dict[Pair, dict[int, int]],
) -> tuple[bytes, ...]:
    if DEBUG:
        print("tokens")
        print(tokens)
        print("pair_max")
        print(pair_max)

    token_mergerd = pair_max[0] + pair_max[1]
    result: list[bytes] = []
    i = 0
    last_token = b""
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair_max:
            result.append(token_mergerd)

            # update freq and appears
            dec_pair(pretok_id, cnt, (tokens[i], tokens[i + 1]), freq, aprs)
            if i > 0:
                dec_pair(pretok_id, cnt, (last_token, tokens[i]), freq, aprs)
                inc_pair(pretok_id, cnt, (last_token, token_mergerd), freq, aprs)
            if i + 2 < len(tokens):
                dec_pair(pretok_id, cnt, (tokens[i + 1], tokens[i + 2]), freq, aprs)
                inc_pair(pretok_id, cnt, (token_mergerd, tokens[i + 2]), freq, aprs)

            last_token = token_mergerd
            i += 2
        else:
            result.append(tokens[i])
            last_token = tokens[i]
            i += 1

    return tuple(result)


def record_pairs(
    pretoks: dict[int, tuple[tuple[bytes, ...], int]],
) -> tuple[
    dict[Pair, int],
    dict[Pair, dict[int, int]],
]:
    freq: dict[Pair, int] = {}
    aprs: dict[Pair, dict[int, int]] = {}

    for pretok_id, (tokens, cnt) in pretoks.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            freq[pair] = freq.get(pair, 0) + cnt
            # add pretok to appears of pair
            aprs_dict: dict[tuple[bytes, ...], int] = aprs.get(pair, {})
            aprs_dict[pretok_id] = aprs_dict.get(pretok_id, 0) + cnt
            aprs[pair] = aprs_dict

    return (freq, aprs)


def merge(
    pretoks: dict[int, tuple[tuple[bytes, ...], int]],
    vocab_size: int,
    vocab: dict[int, bytes],
    show_progress: bool = False,
) -> list[Pair]:
    merges: list[Pair] = []
    freq, aprs = record_pairs(pretoks)
    progress = (
        tqdm(
            total=max(vocab_size - len(vocab), 0),
            desc="Merging pairs",
            unit="merge",
        )
        if show_progress
        else None
    )

    while len(vocab) < vocab_size:
        # find most frequent pair
        pair_max, _ = find_top_freq(freq)

        if DEBUG:
            print("pair_max")
            print(pair_max)

        merges.append(pair_max)
        # vocab part 3: merges
        vocab[len(vocab)] = pair_max[0] + pair_max[1]

        if DEBUG:
            print("pretoks")
            print(pretoks)

        aprs_pretoks = list(aprs.get(pair_max, {}).keys())
        if DEBUG:
            print("aprs_pretoks")
            print(aprs_pretoks)

        for pretok_id in aprs_pretoks:
            tokens, cnt = pretoks.pop(pretok_id)
            new_tokens = merge_pairs(pretok_id, tokens, cnt, pair_max, freq, aprs)
            pretoks[pretok_id] = (new_tokens, cnt)

        if progress is not None:
            progress.update()

    if progress is not None:
        progress.close()

    return merges


def tokenization(
    chunk: tuple[int, int], input_path: Path, special_tokens: list[str]
) -> dict[bytes, int]:
    with open(input_path, "rb") as f:
        start, end = chunk
        f.seek(start)
        chunk_bytes = f.read(end - start).decode("utf-8", errors="ignore")

        split_chunks: Iterator[str] = re.splititer(
            "|".join(re.escape(token) for token in special_tokens), chunk_bytes
        )

        pretokens: dict[bytes, int] = {}
        for split_chunk in split_chunks:
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            matches = pretoken_match(split_chunk)
            for match in matches:
                pretoken_str = match.group()
                pretok = pretoken_str.encode("utf-8")
                pretokens[pretok] = pretokens.get(pretok, 0) + 1
        return pretokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
) -> tuple[dict[int, bytes], list[Pair]]:

    with open(input_path, "rb") as f:
        special_tokens_bytes: list[bytes] = [
            special_token.encode("utf-8") for special_token in special_tokens
        ]
        boundaries = find_chunk_boundaries(f, CHUNKS_NUM, special_tokens_bytes)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks: list[tuple[int, int]] = []
        for start, end in itertools.pairwise(boundaries):
            chunks.append((start, end))

        with Pool(processes=WORKERS_NUM) as pool:
            if show_progress:
                tokenize = partial(
                    tokenization, input_path=input_path, special_tokens=special_tokens
                )
                partial_tokens = list(
                    tqdm(
                        pool.imap(tokenize, chunks),
                        total=len(chunks),
                        desc="Pre-tokenizing",
                        unit="chunk",
                    )
                )
            else:
                partial_tokens = pool.starmap(
                    tokenization,
                    zip(
                        chunks,
                        itertools.repeat(input_path),
                        itertools.repeat(special_tokens),
                    ),
                )

        pretoks_dict: dict[bytes, int] = {}
        for partial_result in partial_tokens:
            for tokens, count in partial_result.items():
                pretoks_dict[tokens] = pretoks_dict.get(tokens, 0) + count
        pretoks: dict[int, tuple[tuple[bytes, ...], int]] = {}
        for tokens, cnt in pretoks_dict.items():
            byte_tokens = tuple(bytes([byte]) for byte in tokens)
            pretoks[len(pretoks)] = (byte_tokens, cnt)

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
        merges: list[Pair] = merge(
            pretoks, vocab_size, vocab, show_progress=show_progress
        )

        """
        if DEBUG:
            print(vocab)
            print(merges)
        """

        return (vocab, merges)


"""
uv run cs336_basics/train_bpe.py --input-path tests/fixtures/corpus.en --vocab-size 500 --special-token '<|endoftext|>'
uv run cs336_basics/train_bpe.py --input-path tests/fixtures/tinystories_sample.txt --vocab-size 500 --special-token '<|endoftext|>' --output-path outputs/tinystories
uv run cs336_basics/train_bpe.py --input-path tests/fixtures/tinystories_sample_5M.txt --vocab-size 10000 --special-token '<|endoftext|>'
"""


def write_bpe_outputs(
    vocab: dict[int, bytes],
    merges: list[Pair],
    output_path: str | os.PathLike,
) -> None:
    """Write one byte-exact Python tuple per line for readable artifacts."""
    vocab_data = "\n".join(repr(item) for item in sorted(vocab.items())) + "\n"
    merges_data = "\n".join(repr(item) for item in merges) + "\n"

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "vocab.txt").write_text(vocab_data, encoding="utf-8")
    (output_path / "merges.txt").write_text(merges_data, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the training corpus.",
    )
    parser.add_argument(
        "--vocab-size", type=int, required=True, help="Total vocabulary size."
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="A special token. Repeat this argument to provide multiple special tokens",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs"),
        help="Directory for lossless vocab.txt and merges.txt artifacts (default: outputs/).",
    )
    args = parser.parse_args()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        show_progress=True,
    )
    write_bpe_outputs(vocab, merges, args.output_path)


if __name__ == "__main__":
    main()
