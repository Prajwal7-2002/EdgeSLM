# Contrast set

`contrast_test.jsonl` holds 40 minimal pairs (80 documents, 20 per label).
The two documents in a pair share topic and vocabulary but get different
labels:

- a help article explaining password resets (PUBLIC) next to a handover note
  that contains a password (RESTRICTED)
- a press release with a revenue figure (PUBLIC) next to a board pre-read with
  the same figure under embargo (CONFIDENTIAL)
- a wiki page saying "this page contains no secrets" (INTERNAL) next to the
  output of a key rotation job (RESTRICTED)

Several RESTRICTED documents contain a bare secret (a connection string, a key
ID, a recovery code) with no word like "password" anywhere. A system that
classifies by vocabulary should do badly here, and it does: TF-IDF on the v1
training data scores 47.5%.

The metric that matters most on this set is **pair accuracy**: a pair counts
only when both documents are right, since a keyword model often gets one side
of a pair right by accident.

## Provenance and its limits

- Written by Claude (Anthropic) in one session, from the four label definitions
  in `src/inference/prompt.py`. Labels are the author's. **No human has
  reviewed them.**
- Frozen before the v2 adapter was trained. `src/evaluation/datasets.py` pins
  the file's SHA-256, so any edit after the fact, including relabelling rows a
  model gets wrong, makes the evaluation refuse to run.
- The same author then wrote `data/generate_dataset_v2.py`. A first draft of
  that generator paraphrased many contrast documents. That draft was thrown
  away before any training, and the generator was rewritten around situations
  this set does not use. `src/evaluation/audit_overlap.py` shows what remains:
  the closest training document to any contrast document has a TF-IDF cosine
  of about 0.4, lower than for the older `realistic` set, and no 8-word phrase
  is shared. Still, one author wrote both, so the concepts overlap by
  construction. Treat scores here as **optimistic for v2**, not as a blind test.

## Making it a real blind test

The strongest version of this benchmark is pairs written by someone who has not
seen the training generator. To add some, write them to a new file in the same
format (`id`, `pair`, `label`, `text`), register it in
`src/evaluation/datasets.py` with its own hash, and run the ablation. Twenty
pairs written by a person would be worth more than everything in this file.
