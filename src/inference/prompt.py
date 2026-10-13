"""The prompt and label set shared by training, evaluation and the API.

Keeping one copy matters: the adapter only works with the exact prompt it was
trained on, and the label scorer depends on each label's first token.
"""

LABELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")

CLASSIFICATION_GUIDANCE = (
    "PUBLIC = approved information intended for external sharing.\n"
    "INTERNAL = routine company-only procedures or workflows.\n"
    "CONFIDENTIAL = private business, employee, customer, or contract information.\n"
    "RESTRICTED = credentials, keys, tokens, passwords, or access-bypass instructions."
)


def build_prompt(text, guidance=True):
    if guidance:
        return (
            "Classify the sensitivity of the following company document.\n\n"
            f"Use these definitions:\n{CLASSIFICATION_GUIDANCE}\n\n"
            f"Document:\n{text}\n\n"
            "Classification:"
        )
    return (
        "Classify the sensitivity of the following company document.\n\n"
        f"Document:\n{text}\n\n"
        "Classification:"
    )


def completion(label):
    return " " + label


def label_token_ids(tokenizer):
    """First token of each label as it follows the prompt.

    Scoring compares the model's next-token logits at these four ids, so they
    must be distinct; a tokenizer that merges two labels' first tokens would
    make the classifier unable to tell them apart.
    """
    ids = {
        label: tokenizer(completion(label), add_special_tokens=False)["input_ids"][0]
        for label in LABELS
    }
    if len(set(ids.values())) != len(LABELS):
        raise ValueError(f"labels share a first token under this tokenizer: {ids}")
    return ids
