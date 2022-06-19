import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, RobertaConfig

from hf_token_class.pipeline import get_processed_examples
from hf_token_class.roberta_multi_token import RobertaForMultiTokenClassification
from tqdm.auto import tqdm

examples = get_processed_examples()

num_epochs = 3
batch_size = 2


config = RobertaConfig(num_labels=4, return_dict=False)

model = RobertaForMultiTokenClassification.from_pretrained(
    pretrained_model_name_or_path="roberta-base"
)
model.set_classification_head(768, 4)

dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model.forward(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import ipdb

ipdb.set_trace()

# TODO:
# Load the model
# Test out predictions on some sample data
# to see if the model was successfully learning


# if we have input tensors of size
# (1, n_sequence, n_classes)
# then we usually want to softmax along dim=2, if we want to know which class is mot probably

from transformers import RobertaTokenizerFast
from torch.nn.functional import softmax

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
sample_text = "Filer's name is: Jerry Genser but not Somjai Genser"
encoded = tokenizer(sample_text)

input_ids = torch.tensor([encoded.input_ids])
attention_mask = torch.tensor([encoded.attention_mask])


output = model(
    input_ids=torch.unsqueeze(examples[0]["input_ids"], 0).to(device),
    attention_mask=torch.unsqueeze(examples[0]["attention_mask"], 0).to(device),
)

