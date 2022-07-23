import typing as t
import torch
from torch.utils.data import DataLoader
from common.finetune.roberta import RobertaForMultiTokenClassification
from datasets import ClassLabel
from transformers import AdamW, RobertaConfig, get_scheduler
from tqdm.auto import tqdm


class TokenClassificationModel:
    def __init__(
        self,
        class_label: ClassLabel,
        num_epochs: int = 5,
        batch_size: int = 2,
        num_warmup_steps: int = 10,
        learning_rate: float = 0.00005,
    ):
        self.class_label = class_label
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_warmup_steps = num_warmup_steps
        self.learning_rate = learning_rate
        self.model = self.init_model()
        self.config = self.init_config()
        self.device = self.init_device()

    @classmethod
    def from_saved(cls, path: str):
        # TODO: This method will initialize a finetuned model
        # from a saved checkpoint on disk
        pass

    def init_device(self):
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "Cuda is not detected by torch, don't bother training on cpu"
            )
        return torch.device("cuda")

    def init_model(self, path: str = None):
        raise NotImplementedError("Implemented by subclass")

    def init_config(self):
        raise NotImplementedError("Implemented by subclass")

    def finetune(self, examples: t.List[t.Dict]):
        """
        Finetune the model with examples
        """
        # Iniitalize objects for training
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        dataloader = DataLoader(examples, shuffle=True, batch_size=self.batch_size)
        num_training_steps = self.num_epochs * len(dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Run the training loop
        # TODO: Add diagnostics to show loss when training
        self.model.to(self.device)
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model.forward(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        # TODO: After training loop is over, make sure to free model from device

    def predict(self, examples: t.List[str]):
        # This will predict on `examples` using the finetune
        # model.

        # TODO: Implement a way to set model to device
        # and then remove it once predictions are completed
        pass


class RobertaTokenClassFinetune(TokenClassificationModel):
    def init_model(self, path: str = None):
        if path:
            # TODO: Read model file from dict,
            # would be for running predict on
            # a finetuned model
            pass

        model = RobertaForMultiTokenClassification.from_pretrained(
            pretrained_model_name_or_path="roberta-base"
        )
        model.set_classification_head(768, self.class_label.num_classes)
        return model

    def init_config(self):
        return RobertaConfig(num_labels=self.class_label.num_classes, return_dict=False)
