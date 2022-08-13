import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
import json
import os
import numpy as np
from sam import SAM

class MultiSetTrainer:

    def __init__(self,
                model,
                init_train_set,
                train_set,
                eval_set,
                test_set,
                device,
                results_dir='out'):

        self.model = model
        self.init_train_set=init_train_set
        self.train_set=train_set
        self.eval_set=eval_set
        self.test_set=test_set
        self.results_dir=results_dir
        self.device = device

    def train_model(self,
                num_epochs=3,
                learning_rate=5e-5,
                optimizer_name,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8):

        results = {}

        init_dataloader = DataLoader(self.init_train_set, shuffle=True, batch_size=per_device_train_batch_size)
        train_dataloader = DataLoader(self.train_set, shuffle=True, batch_size=per_device_train_batch_size)
        eval_dataloader = DataLoader(self.eval_set, batch_size=per_device_eval_batch_size)
        test_dataloader = DataLoader(self.test_set, batch_size=per_device_eval_batch_size)

        if optimizer_name == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        elif optimizer_name == 'Sam':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(self.model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

        num_training_steps = len(init_dataloader) + (num_epochs * len(train_dataloader))
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for batch in init_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model.training_step(**batch)
            loss.backward()

            if optimizer_name == 'Sam':
                optimizer.first_step(zero_grad=True)
                # second forward-backward pass
                self.model.training_step(**batch).backward()  # make sure to do a full forward pass
                optimizer.second_step(zero_grad=True)

            else:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model.training_step(**batch)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            metric = load_metric("accuracy")
            self.model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = self.model.forward(**batch)
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            results[f'epoch {epoch}'] = metric.compute()

            with open(f'{self.results_dir}/predictions_{epoch}.txt', 'w') as json_file:
                predictions_lst = []
                for batch in test_dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.no_grad():
                        logits = self.model.forward(**batch)
                    predictions = torch.argmax(logits, dim=-1)
                    for pred in predictions:
                        predictions_lst.append({'index': len(predictions_lst), 'prediction': int(pred)})
                for prediction in predictions_lst:
                    json.dump(prediction, json_file)
                    if prediction != predictions_lst[-1]:
                        json_file.write('\n')

        with open(f'{self.results_dir}/results.json', 'w') as outfile:
            json.dump(results, outfile)
