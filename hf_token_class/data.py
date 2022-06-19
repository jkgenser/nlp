import typing as t
import json
from datasets import Dataset
import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data"
ORG_NAME = "org-annotate.csv"
DISCLOSURE_NAME = "twenty_annotations_dataset_single_rows.csv"


def get_disclosures():
    raw = pd.read_csv(str(DATA_PATH / DISCLOSURE_NAME))
    raw["text_labels"] = raw["annotation"].apply(json.loads)
    return Dataset.from_pandas(raw)


def get_classes_from_label_column(rows: t.List[t.Dict]) -> t.Dict[str, int]:
    class_names = set()
    for row in rows:
        for label in row:
            class_names.add(label["label"])
    return {class_name: idx for idx, class_name in enumerate(sorted(list(class_names)))}


if __name__ == "__main__":
    disclosures = get_disclosures()
