import torch
from projects.hf_token_class.data import get_org
from transformers import RobertaTokenizerFast
from common.pipeline import Pipeline, TokenClassificationPipeline
from common.labels import class_label_for_ds


def test_old_pipeline():
    ds = get_org()
    # ds = get_disclosures()
    class_label = class_label_for_ds(ds)
    pipeline = Pipeline(
        tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"),
        class_label=class_label,
    )
    examples = pipeline.prepare_for_training(ds)


def test_token_class_pipeline_roberta():
    ds = get_org()
    class_label = class_label_for_ds(ds)
    pipeline = TokenClassificationPipeline(class_label=class_label)
    examples = pipeline.process_dataset_for_training(ds)

    # Check that we have padding correctly
    assert torch.all(
        torch.eq(
            examples[0]["input_ids"][:25],
            torch.tensor(
                [
                    0,
                    2571,
                    947,
                    565,
                    2288,
                    11,
                    1431,
                    7,
                    907,
                    24182,
                    438,
                    2915,
                    13,
                    55,
                    87,
                    68,
                    1096,
                    325,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            ),
        )
    )

    assert torch.all(
        torch.eq(
            examples[0]["attention_mask"][:25],
            torch.tensor(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
        )
    )

    assert torch.all(
        torch.eq(
            examples[0]["labels"][:25],
            torch.tensor(
                [
                    [0.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ]
            ),
        )
    )


def test_token_class_pipeline_chunked_roberta():
    # TODO: This test will use a smaller input size and check
    # that chunking is working correctly, at least for

    pass
