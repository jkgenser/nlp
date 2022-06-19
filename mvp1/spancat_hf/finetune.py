import random
from collections import defaultdict
import typing as t
import pandas as pd
from regex import P
import torch
from torch.optim import AdamW
from transformers import (
    PreTrainedTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
    get_scheduler,
)
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_metric
from mvp1.spancat_hf.prepreocess import get_financial_disclosure, get_processed_org
from mvp1.spancat_hf.roberta_multi_token import RobertaForMultiTokenClassification


def get_label_mapping(data: pd.DataFrame, label_col_name: str):
    """
    Get mapping of idx to label
    """
    labels_set = set()
    for labels in data[label_col_name]:
        for label in labels:
            labels_set.add(label["label"])
    return {idx: e for e, idx in enumerate(sorted(list(labels_set)))}


def overlaps(seq1: t.Tuple[int, int], seq2: t.Tuple[int, int]):
    s1, e1, s2, e2 = seq1[0], seq1[1], seq2[0], seq2[1]

    # endpoints touching
    if s1 == s2:
        return True
    if s1 == e2:
        return True
    if e1 == s2:
        return True
    if e1 == e2:
        return True

    # s1 contained in seq2
    if s2 <= s1 <= e2:
        return True

    # e2 contained in seq2
    if s2 <= e1 <= e2:
        return True

    # seq2 completely contained in seq1
    if s1 <= s2 and e1 >= e2:
        return True

    return False


def split_long(
    be: BatchEncoding,
    input_size: int = 512,  # roberta
):
    """
    If our tokenized text is longer than the input of the model, then we need
    to split into 510 length, and then add back separators [CLS] and [SEP] to
    the begining and end of each input so that it is the same as the origianl.

    labeled_token_offsets: {
        "org": [1, 2, 3]
        "foo": [3, 4, 5]
    }
    """

    # Early exit if we have a sequence that has padding
    # we infer if there is padding based on if the tokenizer
    # ever set attention_mask to 0
    if sum(be.attention_mask) <= input_size:
        return [
            {
                "input_ids": be.input_ids,
                "attention_mask": be.attention_mask,
                "offset_mapping": be.offset_mapping,
            }
        ]

    to_split_len = input_size - 2  # [CLS] and [SEP] and beginning and end

    # Generate indices for each of the input examples
    split_idxs = [
        (i, i + to_split_len) for i in range(0, len(be.input_ids), to_split_len)
    ]

    # For each index, we need to generate a new BatchEncoding that is mapped to the new indices
    encoded_batches = []
    for start, end in split_idxs:
        encoded = {}
        # We add back padding here
        # [CLS] is 0 [SEP] is 2
        # These are hard-coded to roberta values
        encoded["input_ids"] = [0] + be.input_ids[start:end] + [2]
        encoded["attention_mask"] = [1] + be.attention_mask[start:end] + [1]
        encoded["offset_mapping"] = [(0, 0)] + be.offset_mapping[start:end] + [(0, 0)]
        encoded_batches.append(encoded)

    return encoded_batches


def labels_for_encoding(encoded: t.Dict, labels: t.List[t.Dict]):
    """
    Get labels for this batch encoded text
    Note, offset_mapping is globally indexed into the original text

    encoded: {input_ids, attention, offset_mapping}
    """
    labeled_token_offsets = defaultdict(list)
    for idx, offset in enumerate(encoded["offset_mapping"]):
        if offset == (0, 0):
            continue
        for label in labels:
            class_name = label["label"]
            label_tup = (label["start"], label["end"])

            if overlaps(label_tup, offset):
                labeled_token_offsets[class_name].append(idx)

    return labeled_token_offsets


def input_to_tensors(input):
    # generate multi-hot encoded labels tensor
    input_size = len(input["input_ids"])

    # generate base tensor for labels
    # dimension (input_size x num_labels)
    labels_tensor = torch.zeros(input_size, len(LABEL2IDX))
    label_idxs = input["labeled_token_idxs"]

    # For each class, set one-hot encoding
    for class_ in label_idxs.keys():
        token_idxs = label_idxs[class_]
        class_idx = LABEL2IDX[class_]
        for tok_idx in token_idxs:
            labels_tensor[tok_idx, class_idx] = 1

    return {
        "input_ids": torch.tensor([input["input_ids"]]),
        "attention_mask": torch.tensor([input["attention_mask"]]),
        "labels": labels_tensor,
    }


def pipeline(
    tokenizer: PreTrainedTokenizer,
    text: str,
    labels: t.List[str],
    input_size: int = 512,
):
    be: BatchEncoding = tokenizer(
        text,
        return_offsets_mapping=True,
        padding="max_length",
        add_special_tokens=False,  # We will split later on and add [CLS] and [SEP] tokens
    )

    encoded_chunks = split_long(be, input_size=input_size)

    for chunk in encoded_chunks:
        chunk["labeled_token_idxs"] = labels_for_encoding(chunk, labels)

    inputs = [input_to_tensors(chunk) for chunk in encoded_chunks]

    return inputs


data, label_col = get_financial_disclosure(), "labels"
# data, label_col = get_processed_org(), "labels"


test_text = data["text"][0]  # this is a long text with multiple labels
test_labels = data["labels"][0]


# Create a tokenizer instance that was used to train roberta-base
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# encoding_with_labels = pipeline(tokenizer, test_text, test_labels)
LABEL2IDX = get_label_mapping(data, label_col)
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# Training configuration
model = RobertaForMultiTokenClassification(
    RobertaConfig(num_labels=len(LABEL2IDX), return_dict=False)
)


optimizer = AdamW(model.parameters(), lr=0.00005)
num_epochs = 4
batch_size = 2
num_training_steps = num_epochs * len(data)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Create training data
training_data = [pipeline(tokenizer, row.text, row.labels) for row in data.itertuples()]
training_data_flattened = [td for tdl in training_data for td in tdl]


import ipdb

ipdb.set_trace()

# Training loop
for epoch in range(num_epochs):
    # Shuffle training data on each epoch
    random.shuffle(training_data_flattened)
    split_idxs = [
        (i, i + batch_size) for i in range(0, len(training_data_flattened), batch_size)
    ]
    batch = training_data_flattened[split_idxs[0] : split_idxs[1]]


# Metrics loop
# TODO:
# metric = load_metric("")

import ipdb

ipdb.set_trace()


# Heuristics --
# Take start and end, any tokens that contain start an end are assigned to the label
# once they are assigned, can't be re-assigned

# tokenize(return_offsets_mapping=True)
# return_offsets_mapping --> this is how map indices from input characters to tokenized characters

# for label in labels:
#   for token in tokens:
#       if overlaps(label, token):
# assing label to token at corresponding index

########### ##############
#         #
####  ###  ####  #### ###


# def tokenize_and_map_labels(
#     tokenizer: PreTrainedTokenizer,
#     text: str,
#     labels: t.List[str],
#     input_size: int = 512,
# ):
#     """


#     text: document text
#     labels: list of labels and offsets for each text
#     """
#     labeled_token_offsets = defaultdict(list)

#     # token batch
#     be: BatchEncoding = tokenizer(
#         text,
#         return_offsets_mapping=True,
#         padding="max_length",
#         add_special_tokens=False,  # We will split later on and add [CLS] and [SEP] tokens
#     )

#     # Perform one pass over the list of token offsets, and determine
#     # the token is part of one of our sequence labels
#     for idx, offset in enumerate(be.offset_mapping):
#         # Early exit if we hit a special token with offset (0, 0)
#         if offset == (0, 0):
#             continue
#         for label in labels:
#             class_name = label["label"]
#             label_tup = (label["start"], label["end"])

#             if overlaps(label_tup, offset):
#                 labeled_token_offsets[class_name].append(idx)

#     return be, labeled_token_offsets
