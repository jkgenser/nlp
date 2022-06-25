import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, sigmoid
from transformers import AdamW, get_scheduler, RobertaConfig, RobertaTokenizerFast

from hf_token_class.pipeline import Pipeline, class_label_for_ds
from hf_token_class.data import get_org, get_disclosures
from hf_token_class.roberta_multi_token import RobertaForMultiTokenClassification
from tqdm.auto import tqdm

torch.set_printoptions(sci_mode=False)

# ds = get_org()
ds = get_disclosures()
class_label = class_label_for_ds(ds)
pipeline = Pipeline(
    tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"),
    class_label=class_label,
)
examples = pipeline.prepare_for_training(ds)


config = RobertaConfig(num_labels=4, return_dict=False)

model = RobertaForMultiTokenClassification.from_pretrained(
    pretrained_model_name_or_path="roberta-base"
)
model.set_classification_head(768, pipeline.class_label.num_classes)


num_epochs = 20
batch_size = 2

dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
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
    # return sigmoid(outputs.logits)
    return softmax(outputs.logits, dim=-1)


# We need to scale by sigmoid when we have multiclass problem
# We need to scale by soiftmax in the special case where we are
# only doing binary classification


[0, 0, 0, 0]
[0.04, 0.05, 0.13, 0.5]
# np.where(>0.5)


# training in fewer epochs
# reward the model more for making positive
# predictions than making "background class"
# predictions
