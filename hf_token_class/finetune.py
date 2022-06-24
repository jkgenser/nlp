import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, RobertaConfig

from hf_token_class.pipeline import get_processed_examples
from hf_token_class.roberta_multi_token import RobertaForMultiTokenClassification
from tqdm.auto import tqdm

examples = get_processed_examples()

num_epochs = 500
batch_size = 2
N_CLASSES = 1  # or 4


config = RobertaConfig(num_labels=4, return_dict=False)

model = RobertaForMultiTokenClassification.from_pretrained(
    pretrained_model_name_or_path="roberta-base"
)
model.set_classification_head(768, N_CLASSES)

# dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
dataloader = DataLoader([examples[0]], shuffle=True, batch_size=1)
optimizer = AdamW(model.parameters(), lr=0.00005)
num_training_steps = num_epochs * len(dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))
loss_per_batches = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model.forward(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        running_loss += loss.item()

        print(f"loss: {loss}")

        # if i % loss_per_batches == 0:
        #     print(f"batch: {i}, loss: {running_loss / loss_per_batches}")
        #     running_loss = 0

# How much should loss be going down?
# Do I need to retry on a different dataset that is simpler... like org annotate?

# TODO:
# Load the model
# Test out predictions on some sample data
# to see if the model was successfully learning


# if we have input tensors of size
# (1, n_sequence, n_classes)
# then we usually want to softmax along dim=2, if we want to know which class is mot probably

from transformers import RobertaTokenizerFast
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
sample_text = "Will Chris get fired from Commonwealth Care Alliance?"
encoded = tokenizer(sample_text)
input_ids = torch.tensor([encoded.input_ids])
attention_mask = torch.tensor([encoded.attention_mask])


output = model(
    input_ids=torch.unsqueeze(examples[0]["input_ids"], 0).to(device),
    attention_mask=torch.unsqueeze(examples[0]["attention_mask"], 0).to(device),
)


sample_text = "AT&T reportedly in talks to buy DirecTV for more than $50 billion"
encoded = tokenizer(sample_text)
input_ids = torch.tensor([encoded.input_ids])
attention_mask = torch.tensor([encoded.attention_mask])


def predict_on_example(sample_text: str):
    encoded = tokenizer(sample_text)
    input_ids = torch.tensor([encoded.input_ids])
    attention_mask = torch.tensor([encoded.attention_mask])
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    return sigmoid(outputs.logits)


# We need to scale by sigmoid when we have multiclass problem
# We need to scale by soiftmax in the special case where we are
# only doing binary classification

import ipdb

ipdb.set_trace()

[0, 0, 0, 0]
[0.04, 0.05, 0.13, 0.5]
# np.where(>0.5)


# training in fewer epochs
# reward the model more for making positive
# predictions than making "background class"
# predictions
