# Assignment 1 (basics): Building a Transformer LM

## Part 1 - Byte-Pair Encoding (BPE) Tokenizer

### train_bpe_tinystories (a)

I trained a byte-level BPE tokenizer on the TinyStories training dataset with a
maximum vocabulary size of 10,000, including the `<|endoftext|>` special token.
The resulting vocabulary and merges were serialized to disk for inspection.

#### Results

| Metric | Result |
| --- | --- |
| Vocabulary size | 10,000 tokens |
| Number of merges | 9,743 |
| Training time | 261.52 seconds (approximately 4 minutes 22 seconds) |
| Longest token | `b' accomplishment'` |
| Longest-token length | 15 bytes |
| Peak memory usage | Not recorded |
