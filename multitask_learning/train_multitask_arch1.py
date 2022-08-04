import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
import json

NUM_LABELS = 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CausalityModel(transformers.PreTrainedModel):
    def __init__(self, entailment_model, event_detection_model, tokenizer):
        super().__init__(transformers.PretrainedConfig())
        self.entailment_model = entailment_model
        self.event_detection_model = event_detection_model
        self.tokenizer = tokenizer
        self.fc_out = nn.Linear(NUM_LABELS, NUM_LABELS)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    @classmethod
    def create(cls, path_to_entailment, path_to_event_detection, path_to_tokenizer):

        entailment_model = AutoModelForSequenceClassification.from_pretrained(path_to_entailment)
        event_detection_model = AutoModelForSequenceClassification.from_pretrained(path_to_event_detection)
        tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
        attr_name = cls.get_encoder_attr_name(entailment_model)
        shared_encoder = getattr(entailment_model, attr_name)
        setattr(event_detection_model, attr_name, shared_encoder)
        return cls(entailment_model=entailment_model, event_detection_model=event_detection_model, tokenizer=tokenizer)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Electra"):
            return "electra"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, **kwargs):
        entailment_logits = self.entailment_model(**kwargs).logits
        event_detection_logits = self.event_detection_model(**kwargs).logits
        out = self.fc_out(entailment_logits + event_detection_logits)
        return out

    def training_step(self, **kwargs):
        logits = self.forward(**kwargs)
        loss = self.loss(logits, kwargs['labels']).mean()
        return loss

    @staticmethod
    def load_csv(file):
        dataset = load_dataset('csv', data_files=file)
        dataset = dataset['train']
        return dataset

    @staticmethod
    def tokenize_data(dataset, tokenizer):
        #tokenize data
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def set_datasets(self, train_set, test_set):
        self.tokenized_train_set = self.tokenize_data(self.load_csv(train_set), self.tokenizer)
        self.tokenized_dev_set = self.tokenize_data(self.load_csv(test_set), self.tokenizer)

    def train_model(self,
                num_epochs,
                learning_rate,
                per_device_train_batch_size,
                per_device_eval_batch_size,
                results_file):

        results = {}
        train_dataloader = DataLoader(self.tokenized_train_set, shuffle=True, batch_size=per_device_train_batch_size)
        eval_dataloader = DataLoader(self.tokenized_dev_set, batch_size=per_device_eval_batch_size)

        optimizer = AdamW(self.parameters(), lr=learning_rate)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        self.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = self.training_step(**batch)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            metric = load_metric("accuracy")
            self.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = self.forward(**batch)

                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            results[f'epoch {epoch}'] = metric.compute()

        with open(f'{results_file}.json', 'w') as outfile:
            json.dump(results, outfile)


def main():
    parser = argparse.ArgumentParser(description = "train and evaluate multi-task learning model")
    parser.add_argument("-entailment_detection_model",
                        help = "Path to pre-trained entailment detection model",
                        required = True,
                        type = str)
    parser.add_argument("-event_detection_model",
                        help = "Path to pre-trained event detection model",
                        required = True,
                        type = str)
    parser.add_argument("-tokenizer",
                        help = "Path to tokenizer",
                        required = True,
                        type = str)
    parser.add_argument("-train_set",
                        help = ".csv file for training",
                        required = True,
                        type = str)
    parser.add_argument("-test_set",
                        help = ".csv file for evaluation",
                        required = True,
                        type = str)
    parser.add_argument("-num_epochs",
                        help = "number of epochs for training",
                        required = False,
                        default=3,
                        type = int)
    parser.add_argument("-learning_rate",
                        help = "learning rate for training",
                        required = False,
                        default=5e-5,
                        type = float)
    parser.add_argument("-per_device_train_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for training",
                        required = False,
                        default=8,
                        type = int)
    parser.add_argument("-per_device_eval_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for evaluation",
                        required = False,
                        default=8,
                        type = int)
    parser.add_argument("-results_file",
                        help = "file to save results",
                        required = False,
                        default='results',
                        type = str)
    args = parser.parse_args()

    causality_model = CausalityModel.create(
        path_to_entailment = args.entailment_detection_model,
        path_to_event_detection = args.event_detection_model,
        path_to_tokenizer = args.tokenizer
    )

    causality_model.set_datasets(train_set=args.train_set, test_set=args.test_set)
    causality_model.to(device)
    training_args = {'num_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'per_device_train_batch_size': args.per_device_train_batch_size,
                    'per_device_eval_batch_size': args.per_device_eval_batch_size,
                    'results_file': args.results_file,
                    }

    causality_model.train_model(**training_args)

if __name__ == '__main__':
    main()
