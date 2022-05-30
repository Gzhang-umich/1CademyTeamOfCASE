import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, load_metric
import argparse

def load_csv(file):
    dataset = load_dataset('csv', data_files=file)
    dataset = dataset['train']
    return dataset

def compute_metrics(eval_pred):
    #accuracy metric
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_data(dataset, tokenizer):
    #tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

class ModelTrainer:


    def __init__(self, model_name, train_set, test_set):
        self.model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenized_train_set = tokenize_data(train_set, self.tokenizer)
        self.tokenized_test_set = tokenize_data(test_set, self.tokenizer)
        #set training args
        self.training_args = TrainingArguments(output_dir= self.model+"_trainer", evaluation_strategy="epoch")
        #initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=2)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        #initialize trainer
        self.trainer = Trainer(
          model=self.model,
          args=self.training_args,
          train_dataset=self.tokenized_train_set,
          eval_dataset=self.tokenized_test_set,
          compute_metrics=compute_metrics
        )

    def train_model(self):
        #train model
        self.trainer.train()

def main():
    parser = argparse.ArgumentParser(description = "train and evaluate PTMs")
    parser.add_argument("-model",
                        help = "name or path to pre-trained model",
                        required = True,
                        type = str)
    parser.add_argument("-train_set",
                        help = ".csv dataset to used for training",
                        required = True,
                        type = str)

    parser.add_argument("-test_set",
                        help = ".csv dataset to used for training",
                        required = True,
                        type = str)
    args = parser.parse_args()

    train_set = load_csv(args.train_set)
    test_set = load_csv(args.test_set)

    modelTrainer = ModelTrainer(args.model, train_set, test_set)
    modelTrainer.train_model()

if __name__=='__main__':
    main()
