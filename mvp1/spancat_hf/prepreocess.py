import pandas as pd
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
ORG_NAME = "org-annotate.csv"
DISCLOSURE_NAME = "twenty_annotations_dataset_single_rows.csv"


def process_labels(label):
    objs = json.loads(label)
    return [
        {
            "start": int(obj["startOffset"]),
            "end": int(obj["endOffset"]),
            "label": obj["label"],
        }
        for obj in objs
    ]


def get_processed_org():
    raw = pd.read_csv(str(DATA_PATH / ORG_NAME))
    raw["labels"] = raw["labels"].apply(process_labels)
    return raw


def get_financial_disclosure():
    raw = pd.read_csv(str(DATA_PATH / DISCLOSURE_NAME))
    raw["labels"] = raw["annotation"].apply(json.loads)
    return raw
