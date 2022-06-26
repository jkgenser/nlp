"""
Single script for fine-tuning text classification using Bio_ClinicalBERT
for sequence classification
"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets.features import ClassLabel
from hf_token_class.pipeline import Chunker
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
raw = pd.read_csv("/home/jerry/nlp/data/mtsamples.csv")

# remove blank transcriptions entirely
raw = raw[~pd.isna(raw["transcription"])]

# get class label obj and apply to dataset
class_label = ClassLabel(names=raw["medical_specialty"].unique())
raw["labels"] = raw["medical_specialty"].apply(class_label.str2int)

# We need to chunk examples because some of them have long tokens
chunker = Chunker(tokenizer)
chunked_examples = []

progress_bar = tqdm(range(len(raw)))
for idx, (example, label) in enumerate(zip(raw["transcription"], raw["labels"])):
    chunks = chunker.chunk(example)
    for chunk in chunks:
        chunked_examples.append(
            {
                "idx": idx,
                "input_ids": chunk["input_ids"],
                "attention_mask": chunk["attention_mask"],
                "label": label,
            }
        )
    progress_bar.update(1)


# TODO: Something is wrong with my examples
# Need to figure out how I am feeding in a 515 length
# vector in some case... bug with chunker?

# split data using an eval set
X_train, X_test, y_train, y_test = train_test_split(
    chunked_examples, [e["label"] for e in chunked_examples]
)


model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=class_label.num_classes
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=X_train,
    eval_dataset=X_test,
)


trainer.train()
