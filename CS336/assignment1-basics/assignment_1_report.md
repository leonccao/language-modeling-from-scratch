# Assignment 1: Building a Transformer LM

## train_bpe_tinystories (a)

I trained a byte-level BPE tokenizer on TinyStories with a maximum vocabulary
size of 10,000 and the `<|endoftext|>` special token. The vocabulary and merge
rules were serialized to disk for inspection.

| Metric | Result |
| --- | --- |
| Vocabulary size | 10,000 tokens |
| Merges | 9,743 |
| Workers | 8 |
| Wall time | 90.30 seconds |
| User / system CPU time | 388.30 / 12.04 seconds |
| Maximum RSS | 927,186,944 bytes (927 MB or 0.86 GiB) |
| Swaps | 0 |
| Longest token | `b' accomplishment'` (15 bytes) |

I measured time and memory with macOS `/usr/bin/time -l` around the full
training test. Pre-tokenization used multiprocessing; workers read independent
file ranges, counted encoded pre-tokens before converting unique values into
byte tuples, and processed special-token-delimited documents lazily with
`regex.splititer`. These changes reduced wall time from 243.28 to 90.30 seconds
and maximum RSS from 2.13 GB to 927 MB.

## train_bpe_tinystories (b)

Scalene showed that pre-tokenization took the most time. Initially, converting
every pre-token occurrence into a tuple of one-byte `bytes` objects accounted
for approximately 74% of the CPU profile. After deferring that conversion, the
main remaining cost was regex matching and extracting each match: the
`tokenization` function accounted for approximately 68% of the optimized CPU
profile. BPE merging took about 29 seconds and was not the primary bottleneck.
