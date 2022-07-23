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
from common.pipeline import Chunker
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
# chunked_examples = []

# progress_bar = tqdm(range(len(raw)))
# for idx, (example, label) in enumerate(zip(raw["transcription"], raw["labels"])):
#     chunks = chunker.chunk(example)
#     for chunk in chunks:
#         chunked_examples.append(
#             {
#                 "idx": idx,
#                 "input_ids": chunk["input_ids"],
#                 "attention_mask": chunk["attention_mask"],
#                 "label": label,
#             }
#         )
#     progress_bar.update(1)


# We're going to do something very naive for now and just include the first
# 512 tokens and truncate everything else

examples = []
progress_bar = tqdm(range(len(raw)))
for idx, (text, label) in enumerate(zip(raw["transcription"], raw["labels"])):
    te = tokenizer(
        text,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        padding="max_length",
    )
    e = {
        "idx": idx,
        "input_ids": te.input_ids,
        "attention_mask": te.attention_mask,
        "label": label,
    }
    examples.append(e)
    progress_bar.update(1)

# split data using an eval set
X_train, X_test, y_train, y_test = train_test_split(
    examples, [e["label"] for e in examples]
)


model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=class_label.num_classes
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=X_train,
    eval_dataset=X_test,
)


trainer.train()


# This model loss does not appear to actually be going down that much :(
# There is probably too much class imbalance in the dataset
# Model also pretty much always predicts most common class with highest
# probabilityt

import ipdb

ipdb.set_trace()


# TODO: Compare the results of bert truncated 512 with tf-idf gbt
# TODO: Try fine-tuning with roberta
