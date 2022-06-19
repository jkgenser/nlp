import typing as t
import torch

from collections import defaultdict
from hf_token_class.data import get_classes_from_label_column, get_disclosures
from transformers import RobertaTokenizerFast


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


class Pipeline:
    def __init__(self, tokenizer, input_size: int, label2index: t.Dict[str, int]):
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.label2index = label2index
        self.num_labels = len(label2index)

    def chunk_example(self, te):
        """This is hard-coded to roberta logic"""
        if sum(te.attention_mask) <= self.input_size:
            return [te]

        len_example = sum(te.attention_mask)
        chunk_length = self.input_size - 2  # [CLS] and [SEP] back later
        split_idxs = [
            (i, i + chunk_length) for i in range(0, len_example, chunk_length)
        ]

        chunks = []

        for start, end in split_idxs:
            chunk = {
                "input_ids": te.input_ids[start:end],
                "attention_mask": te.attention_mask[start:end],
                "offset_mapping": te.offset_mapping[start:end],
            }

            # Note: logic below is hard coded to roberta
            # We want chunk size of 510 = 512 - 2
            want_chunk_size = self.input_size - 2
            this_chunk_size = len(chunk["input_ids"])

            # For each token under 510, we need to pad to the right the
            # appropriate way so that we end up with all 512 tensors later
            need_padding = want_chunk_size - this_chunk_size

            chunk["input_ids"] = [0] + chunk["input_ids"] + (need_padding * [1]) + [2]
            chunk["a/home/jerry/nlp/ttention_mask"] = (
                [1] + chunk["attention_mask"] + (need_padding * [0]) + [1]
            )
            chunk["offset_mapping"] = (
                [(0, 0)]
                + chunk["offset_mapping"]
                + (need_padding * [(0, 0)])
                + [(0, 0)]
            )

            chunks.append(chunk)
        return chunks

    def text_to_token_labels(self, te, text_labels) -> t.Dict[str, t.List[int]]:
        """
        Given labels from original text, map them to the token
        index based on the text offset for each token
        """
        token_labels = defaultdict(list)
        for idx, offset in enumerate(te["offset_mapping"]):
            if offset == (0, 0):
                continue
            for label in text_labels:
                class_name = label["label"]
                label_tup = (label["start"], label["end"])
                if overlaps(label_tup, offset):
                    token_labels[class_name].append(idx)
        return token_labels

    def multi_hot_encode_token_labels(self, token_idx_labels):
        """
        Generate a mult-hot encoded tensor for each chunk
        """
        shape = (self.input_size, self.num_labels)
        labels_tensor = torch.zeros(*shape)

        for class_ in self.label2index.keys():
            token_idxs = token_idx_labels.get(class_)
            if not token_idxs:
                continue

            class_idx = self.label2index[class_]
            for tok_idx in token_idxs:
                labels_tensor[tok_idx, class_idx] = 1

        return labels_tensor

    def process(self, example: t.Dict):
        # tokenization of the entire example
        tokenized_example = self.tokenizer(
            example["text"],
            return_offsets_mapping=True,
            padding="max_length",
            add_special_tokens=False,
        )

        # split long text into chunks
        chunks = self.chunk_example(tokenized_example)

        # map the start/end of spans to token idxs
        token_idx_labels = [
            self.text_to_token_labels(chunk, example["text_labels"]) for chunk in chunks
        ]

        # generate multi-hot encoded token labels
        label_tensors = [
            self.multi_hot_encode_token_labels(token_idx_label)
            for token_idx_label in token_idx_labels
        ]

        assert len(chunks) == len(token_idx_labels) == len(label_tensors)

        return {
            "input_ids": [c["input_ids"] for c in chunks],
            "attention_mask": [c["attention_mask"] for c in chunks],
            "labels": label_tensors,
            "text_labels": example["text_labels"],
        }


def get_processed_examples():
    disclosures = get_disclosures()
    label2index = get_classes_from_label_column(disclosures["text_labels"])

    pipeline = Pipeline(
        tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"),
        input_size=512,
        label2index=label2index,
    )
    processed = disclosures.map(pipeline.process)
    final = []

    for batch in processed:
        for input_ids, attention_mask, labels in zip(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        ):
            final.append(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels),
                    # "token_type_ids": torch.zeros(
                    #     pipeline.input_size, dtype=torch.long
                    # ),
                }
            )
    return final
