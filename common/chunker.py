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
        raise NotImplementedError("Implemented by subclass")


class RobertaChunker(Chunker):
    """
    Chunking logic specific to roberta
    """

    def init_tokenizer(self) -> PreTrainedTokenizerFast:
        return RobertaTokenizerFast.from_pretrained("roberta-base")

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

        # TODO: Don't do encode/decode strategy, it's actually incorrect.
        # The biggest issue is that when we re-encode, the offsets mapping gets
        # messed up.

        # instead, we need to split using a smaller chunk length, then add
        # any extra special tokens that are needed to the splitted encoded.

        # In the future, we should be able to do fancier stride stuff.

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

        import ipdb

        ipdb.set_trace()

        return chunks


class BertChunker(Chunker):
    """
    Chunking logic specific to bert
    """

    def init_tokenizer():
        pass

    def encode_chunks(self, text: str):
        # NOTE: when we implement this, we need to use a different strategy than
        # we use with roberta because encode(decode()) for bert uncased model results
        # in not recovering the original text exactly.

        # Instead, we need to manually add start/end and sep tokens after splitting
        # the text ourselves. In the future, we should be able to split along a
        # stride so that we can do more sophisticated strategies for long documents.
        raise NotImplementedError("TODO")
