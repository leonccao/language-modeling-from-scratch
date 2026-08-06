import json
import os
import time

import pytest

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

DATA_PATH = FIXTURES_PATH.parent.parent / "data"
OUTPUT_PATH = FIXTURES_PATH.parent.parent / "outputs"
LARGE_DATASETS = {
    "tinystories": DATA_PATH / "TinyStoriesV2-GPT4-train.txt",
    "openwebtext": DATA_PATH / "owt_train.txt",
}
LARGE_VOCAB_SIZE = {
    "tinystories": 10_000,
    "openwebtext": 32_000,
}
LARGE_OUTPUT_PATHS = {
    "tinystories": OUTPUT_PATH / "TinyStories",
    "openwebtext": OUTPUT_PATH / "OpenWebText",
}


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    train_time = end_time - start_time
    assert train_time < 1.5, f"Training time longer than 1.5s limit: {train_time}"


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_writes_outputs(tmp_path):
    run_train_bpe(
        input_path=FIXTURES_PATH / "corpus.en",
        vocab_size=258,
        special_tokens=["<|endoftext|>"],
        output_path=tmp_path,
    )

    assert (tmp_path / "vocab.txt").is_file()
    assert (tmp_path / "merges.txt").is_file()


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )


def test_train_bpe_large_dataset():
    """Train on a full dataset when BPE_DATASET selects a configured corpus.

    Run one of:
    BPE_DATASET=tinystories uv run pytest tests/test_train_bpe.py::test_train_bpe_large_dataset -s
    BPE_DATASET=openwebtext uv run pytest tests/test_train_bpe.py::test_train_bpe_large_dataset -s
    """
    dataset_name = os.environ.get("BPE_DATASET")
    if dataset_name is None:
        pytest.skip("Set BPE_DATASET=tinystories or BPE_DATASET=openwebtext")

    dataset_name = dataset_name.lower()
    if dataset_name not in LARGE_DATASETS:
        pytest.fail(
            f"Unknown BPE_DATASET={dataset_name!r}; "
            f"choose one of {sorted(LARGE_DATASETS)}"
        )

    vocab_size = LARGE_VOCAB_SIZE[dataset_name]
    input_path = LARGE_DATASETS[dataset_name]
    if not input_path.is_file():
        pytest.fail(f"Dataset does not exist: {input_path}")

    special_token = "<|endoftext|>"

    start_time = time.perf_counter()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=[special_token],
        output_path=LARGE_OUTPUT_PATHS[dataset_name],
    )
    train_time = time.perf_counter() - start_time

    longest_token = max(vocab.values(), key=len)

    print(
        f"\n{dataset_name}: trained {len(vocab):,} tokens and "
        f"{len(merges):,} merges in {train_time:.2f} seconds "
        f"with the longest token as {longest_token!r} ({len(longest_token)} bytes)"
    )

    assert len(vocab) == vocab_size
    assert special_token.encode("utf-8") in vocab.values()
    assert len(merges) == vocab_size - 256 - 1
