import typing as t
import torch
from datasets import Dataset, ClassLabel

from collections import defaultdict
from common.labels import overlaps
from common.chunker import BertChunker, Chunker, RobertaChunker
from transformers import (
    PreTrainedTokenizerFast,
)


class TokenClassificationPipeline:
    def __init__(
        self,
        class_label: ClassLabel,
        input_size: int = 512,
        base: str = None,
    ):
        self.class_label = class_label
        self.input_size = input_size

        if not base:
            self.chunker = RobertaChunker()
        if base == "bert":
            self.chunker = BertChunker()

    def process_example_for_training(self, example: t.Dict):
        """
        Process a single example.
        """
        # split long texts into chunks
        chunks = self.chunker.chunk(example["text"])

        # map start/end of spans to token idxs
        token_idx_labels = [
            self.spans_to_token_offsets(chunk["offset_mapping"], example["text_labels"])
            for chunk in chunks
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

    def process_dataset_for_training(self, dataset: Dataset):
        """
        Given a dataset, create examples that are ready for training
        """
        processed = dataset.map(self.process_example_for_training)
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

    def spans_to_token_offsets(
        self, offset_mapping: t.List[t.Tuple], text_labels: t.List[t.Dict[str, int]]
    ) -> t.Dict[str, t.List[int]]:
        """
        Given labels from original text, e.g. {"start": 0, "end": 4},
        map the text offsets to token offsets, e.g. {"label": [1, 2, 3]}
        """
        token_idx_labels = defaultdict(list)
        for idx, offset in enumerate(offset_mapping):
            if offset == (0, 0):
                continue
            for label in text_labels:
                class_name = label["label"]
                label_tup = (label["start"], label["end"])
                if overlaps(label_tup, offset):
                    token_idx_labels[class_name].append(idx)
        return token_idx_labels

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
        self.chunker = Chunker(tokenizer, input_size)

    def chunk_example(self, text: str):
        return self.chunker.chunk(text)

    def text_to_token_labels(
        self, offset_mapping: t.List[t.Tuple], text_labels: t.List[t.Dict[str, int]]
    ) -> t.Dict[str, t.List[int]]:
        """
        Given labels from original text, map them to the token
        index based on the text offset for each token
        """
        token_labels = defaultdict(list)
        for idx, offset in enumerate(offset_mapping):
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
            self.text_to_token_labels(chunk["offset_mapping"], example["text_labels"])
            for chunk in chunks
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

    def token_preds_to_labels(self):
        pass
