import typing as t
import torch
from datasets import Dataset, ClassLabel

from collections import defaultdict
from hf_token_class.data import get_disclosures, get_org
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast


def overlaps(seq1: t.Tuple[int, int], seq2: t.Tuple[int, int]):
    """
    "two spans overlap if both of their starts come before both of their ends"
    """
    start = 0
    end = 1
    return seq1[start] <= seq2[end] and seq2[start] <= seq1[end]


def class_label_for_ds(ds: Dataset, label_col_name: str = "text_labels") -> ClassLabel:
    """
    For sequence labeling we assume that the labels are formatted like
    [{"start": int, "end": int, "label": str}]
    """
    class_set = set()
    for row in ds[label_col_name]:
        for label in row:
            class_set.add(label["label"])
    return ClassLabel(num_classes=len(class_set), names=list(class_set))


class Pipeline:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        class_label: ClassLabel,
        input_size: int = 512,
    ):
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.class_label = class_label

    def chunk_example(self, text: str):
        """This is hard-coded to roberta logic"""
        te = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=self.input_size,
            add_special_tokens=False,
        )
        if sum(te.attention_mask) <= self.input_size:
            # if after tokenizing one example, there are some attention
            # mask of 0, then that means the example doesn't need to be chunked
            # so we go back to using the original tokenizer and allow it to
            # add special tokens for us
            te = self.tokenizer(
                text,
                return_offsets_mapping=True,
                padding="max_length",
                max_length=self.input_size,
                add_special_tokens=True,
            )
            return [
                {
                    "input_ids": te.input_ids,
                    "attention_mask": te.attention_mask,
                    "offset_mapping": te.offset_mapping,
                }
            ]

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
            chunk["attention_mask"] = (
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
        shape = (self.input_size, self.class_label.num_classes)
        labels_tensor = torch.zeros(*shape)

        for class_ in self.class_label.names:
            token_idxs = token_idx_labels.get(class_)
            if not token_idxs:
                continue

            class_idx = self.class_label.str2int(class_)
            for tok_idx in token_idxs:
                labels_tensor[tok_idx, class_idx] = 1

        return labels_tensor

    def process(self, example: t.Dict):
        # split long text into chunks
        chunks = self.chunk_example(example["text"])

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

    def prepare_for_training(self, dataset: Dataset):
        """
        Given a dataset, create examples that are ready for training
        """
        processed = dataset.map(self.process)
        examples = []
        for batch in processed:
            for input_ids, attention_mask, labels in zip(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            ):
                examples.append(
                    {
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.tensor(attention_mask),
                        "labels": torch.tensor(labels),
                    }
                )
        return examples


if __name__ == "__main__":
    ds = get_org()
    class_label = class_label_for_ds(ds)
    pipeline = Pipeline(
        tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"),
        class_label=class_label,
    )
    examples = pipeline.prepare_for_training(ds)
