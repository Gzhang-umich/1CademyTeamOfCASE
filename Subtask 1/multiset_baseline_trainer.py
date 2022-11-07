import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
import json
import os
import numpy as np

class MS_Baseline_Trainer:

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
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8):

        results = []

        init_dataloader = DataLoader(self.init_train_set, shuffle=True, batch_size=per_device_train_batch_size)
        train_dataloader = DataLoader(self.train_set, shuffle=True, batch_size=per_device_train_batch_size)
        if self.eval_set:
            eval_dataloader = DataLoader(self.eval_set, batch_size=per_device_eval_batch_size)
        if self.test_set:
            test_dataloader = DataLoader(self.test_set, batch_size=per_device_eval_batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = len(init_dataloader) + (num_epochs * len(train_dataloader))
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for batch in init_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model(**batch).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if epoch == num_epochs-1:
                if self.eval_set:
                    accuracy = load_metric("accuracy")
                    f1 = load_metric('f1')
                    recall = load_metric('recall')
                    precision = load_metric('precision')
                    mcc = load_metric('matthews_correlation')

                    metrics = {'accuracy':accuracy, 'f1':f1, 'recall':recall, 'precision':precision, 'mcc':mcc}

                    self.model.eval()
                    for batch in eval_dataloader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.no_grad():
                            logits = self.model(**batch).logits
                        predictions = torch.argmax(logits, dim=-1)
                        for metric in metrics:
                            metrics[metric].add_batch(predictions=predictions, references=batch["labels"])
                    for metric in metrics:
                        results.append(metrics[metric].compute())

            if self.test_set:
                with open(f'{self.results_dir}/predictions_{epoch}.txt', 'w') as json_file:
                    predictions_lst = []
                    for batch in test_dataloader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.no_grad():
                            logits = self.model(**batch).logits
                        predictions = torch.argmax(logits, dim=-1)
                        for pred in predictions:
                            predictions_lst.append({'index': len(predictions_lst), 'prediction': int(pred)})
                    for prediction in predictions_lst:
                        json.dump(prediction, json_file)
                        if prediction != predictions_lst[-1]:
                            json_file.write('\n')

        if self.eval_set:
            with open(f'{self.results_dir}/results.json', 'w') as outfile:
                # for metric, value in zip(results, results.values()):
                #     json.dump(metric, outfile)
                #     outfile.write(': ')
                #     json.dump(value, outfile)
                #     outfile.write('\n')
                json.dump(results, outfile)
