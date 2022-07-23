import typing as t
from transformers import (
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
    BertTokenizerFast,
)


class Chunker:
    def __init__(self, input_size: int = 512):
        self.input_size = input_size
        self.tokenizer = self.init_tokenizer()

    def init_tokenizer(self):
        raise NotImplementedError(
            "Use a class that subclasses chunker which initializes a tokenizer"
        )

    def chunk(self, text: str) -> t.Dict:
        te = self.tokenizer(
            text,
            padding="max_length",
            return_offsets_mapping=True,
            max_length=self.input_size,
        )
        # If part of this sequence doesn't need masking
        # then we don't need to chunk at all
        if sum(te.attention_mask) <= self.input_size:
            return self.encode_single(text)
        return self.encode_chunks(text)

    def encode_single(self, text: str):
        """
        Use standard tokenization
        """
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

    def encode_chunks(self, text: str):
        """
        Encode the text into multiple chunks
        """
        te = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.input_size,
            add_special_tokens=False,
        )
        len_example = sum(te.attention_mask)
        chunk_length = self.input_size - 2  # assume start and end token
        split_idxs = [
            (i, i + chunk_length) for i in range(0, len_example, chunk_length)
        ]
        split_input_ids = [te.input_ids[start:end] for start, end in split_idxs]

        chunks = []

        # TODO: We need to go back to manually chunking and adding sep/padding etc
        # because encode(decode()) for uncased models results in not recovering the
        # original text exactly. In particular, encode(decode(input_ids)) from bert tokenizer
        # can result in increasing the number of tokens. We need to make sure each slice of
        # data is 512 tokens.

        # It seems to work OK for roberta for now, but doesn't appear to work with
        # bert uncased

        # It also might be a problem with Wordpiece vs. BPE since I first came across
        # this issue with WordPiece using a different BERT

        # The issue encode(decode(len 510)) can sometimes be more than 512
        for split in split_input_ids:
            te = self.tokenizer(
                self.tokenizer.decode(split),
                padding="max_length",
                max_length=self.input_size,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            chunks.append(te)
            if len(te.input_ids) > 512:
                import ipdb

                ipdb.set_trace()

        return chunks


class RobertaChunker(Chunker):
    """
    Chunking logic specific to roberta
    """

    def init_tokenizer(self) -> PreTrainedTokenizerFast:
        return RobertaTokenizerFast.from_pretrained("roberta-base")


class BertChunker(Chunker):
    """
    Chunking logic specific to bert
    """

    def init_tokenizer():
        pass
