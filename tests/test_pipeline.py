import pytest
import pandas as pd
from projects.hf_token_class.data import get_org
from transformers import RobertaTokenizerFast
from common.pipeline import Pipeline, TokenClassificationPipeline
from common.labels import class_label_for_ds
from datasets import Dataset


def test_old_pipeline():
    ds = get_org()
    # ds = get_disclosures()
    class_label = class_label_for_ds(ds)
    pipeline = Pipeline(
        tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"),
        class_label=class_label,
    )
    examples = pipeline.prepare_for_training(ds)


@pytest.fixture
def org_ds():
    text = "AT&T reportedly in talks to buy DirecTV for more than $50 billion"
    text_labels = [{"start": 0, "end": 4, "label": "org"}]
    df = pd.DataFrame.from_records([{"text": text, "text_labels": text_labels}])
    dataset = Dataset.from_pandas(df)
    class_label = class_label_for_ds(dataset)
    return dataset, class_label


def test_pipeline_roberta(org_ds):
    dataset, class_label = org_ds
    pipeline = TokenClassificationPipeline(class_label=class_label, input_size=25)
    examples = pipeline.process_dataset_for_training(dataset)
    assert examples[0]["input_ids"].tolist() == [
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

    assert examples[0]["attention_mask"].tolist() == [
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

    assert examples[0]["labels"].tolist() == [
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


def test_pipeline_roberta_chunked(org_ds):
    dataset, class_label = org_ds

    # Set smaller input size to test whether encoding + chunking is working
    # properly
    pipeline = TokenClassificationPipeline(class_label=class_label, input_size=10)
    examples = pipeline.process_dataset_for_training(dataset)
    import ipdb

    ipdb.set_trace()
    pass
