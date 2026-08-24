# Independent Review Set

This directory contains candidate examples awaiting human labels.

For each row in `benchmark_queue.jsonl`, a reviewer must fill `reviewed_label` with exactly one of:

- PUBLIC
- INTERNAL
- CONFIDENTIAL
- RESTRICTED

A reviewer should also fill `review_notes` for ambiguous cases and record the reviewer name or ID in `reviewer_id`.

Only after review is complete should the file be converted into an evaluation set. Do not use these rows for training.
