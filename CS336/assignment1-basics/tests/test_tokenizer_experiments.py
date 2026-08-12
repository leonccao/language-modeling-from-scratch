from pathlib import Path

import pytest

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).parent.parent

EXPERIMENTS = [
    pytest.param(
        "TinyStories",
        PROJECT_ROOT / "data/tokenizer_experiments/tinystories",
        PROJECT_ROOT / "outputs/TinyStories",
        id="tinystories",
    ),
    pytest.param(
        "OpenWebText",
        PROJECT_ROOT / "data/tokenizer_experiments/openwebtext",
        PROJECT_ROOT / "outputs/OpenWebText",
        id="openwebtext",
    ),
]


@pytest.mark.parametrize(("corpus_name", "sample_dir", "artifact_dir"), EXPERIMENTS)
def test_sample_compression_ratio(
    corpus_name: str,
    sample_dir: Path,
    artifact_dir: Path,
):
    vocab_path = artifact_dir / "vocab.txt"
    merges_path = artifact_dir / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        pytest.skip(f"Train {corpus_name} to create lossless tokenizer artifacts.")

    sample_paths = sorted(sample_dir.glob("doc_*.txt"))
    assert len(sample_paths) == 10

    try:
        tokenizer = Tokenizer.from_files(
            vocab_path,
            merges_path,
            special_tokens=["<|endoftext|>"],
        )
    except (SyntaxError, TypeError, ValueError) as error:
        pytest.skip(f"Retrain {corpus_name} with lossless tokenizer artifacts: {error}")
    total_bytes = 0
    total_tokens = 0

    print(f"\n{corpus_name} compression results")
    for sample_path in sample_paths:
        document_bytes = sample_path.read_bytes()
        document = document_bytes.decode("utf-8")
        token_ids = tokenizer.encode(document)

        assert token_ids
        assert tokenizer.decode(token_ids) == document

        byte_count = len(document_bytes)
        token_count = len(token_ids)
        total_bytes += byte_count
        total_tokens += token_count
        print(
            f"{sample_path.name}: bytes={byte_count}, tokens={token_count}, "
            f"bytes/token={byte_count / token_count:.4f}"
        )

    assert total_tokens > 0
    print(
        f"aggregate: bytes={total_bytes}, tokens={total_tokens}, "
        f"bytes/token={total_bytes / total_tokens:.4f}"
    )
