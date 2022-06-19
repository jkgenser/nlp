import spacy
from spacy.tokens import DocBin
import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
DATA_FNAME = "org-annotate.csv"
SPAN_KEY = "sc"


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


raw = pd.read_csv(str(DATA_PATH / DATA_FNAME))
raw["labels"] = raw["labels"].apply(process_labels)


nlp = spacy.blank("en")
db = DocBin()
for row in raw.itertuples():
    # construct a document with the row of text
    doc = nlp(row.text)

    # add spans based on the labels we have
    spans = [
        doc.char_span(label["start"], label["end"], label["label"])
        for label in row.labels
    ]

    # save spans to the SPAN_KEY defined in config.cfg
    doc.spans[SPAN_KEY] = spans

    # add the doc to our docbin
    db.add(doc)


# write to docbin to disk
db.to_disk("./train/train.spacy")
