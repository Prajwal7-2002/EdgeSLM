"""Fit documents of any length into the classifier's prompt.

The prompt wraps the document between fixed instructions and a closing
"Classification:" line, and the label is read from the logits right after that
line. Truncating the *prompt* to the model's limit cuts that line off, so the
model scores some mid-document word instead and still returns a confident
looking label. Long documents are therefore split into chunks that each fit
with the full prompt around them, every chunk is classified, and the document
takes the most sensitive chunk's label: one page with a password makes the
whole file RESTRICTED.
"""

from inference.prompt import LABELS, build_prompt


# Least to most sensitive. A document is as sensitive as its worst chunk.
SEVERITY = {label: rank for rank, label in enumerate(LABELS)}

MAX_PROMPT_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 32


def document_token_budget(tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS, guidance=True):
    """How many document tokens fit once the fixed prompt text is counted."""
    overhead = len(tokenizer(build_prompt("", guidance))["input_ids"])
    # A few tokens of slack: joining text and prompt can tokenise slightly
    # differently at the seams than the two pieces did separately.
    budget = max_prompt_tokens - overhead - 8
    if budget < 32:
        raise ValueError(f"prompt overhead {overhead} leaves no room for a document")
    return budget


def split_into_chunks(tokenizer, text, budget, overlap=CHUNK_OVERLAP_TOKENS):
    """Split text into pieces of at most ``budget`` tokens.

    Neighbouring chunks share ``overlap`` tokens so a secret that straddles a
    boundary still appears whole in one of them. Short documents come back as
    a single chunk identical to the input, so they are classified exactly as
    in training.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return [text]
    if overlap >= budget:
        raise ValueError("overlap must be smaller than the chunk budget")

    chunks = []
    step = budget - overlap
    for start in range(0, len(ids), step):
        piece = ids[start : start + budget]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
        if start + budget >= len(ids):
            break
    return [chunk for chunk in chunks if chunk]


def most_sensitive(labels):
    return max(labels, key=lambda label: SEVERITY[label])
