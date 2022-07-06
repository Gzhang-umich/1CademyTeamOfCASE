import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, load_metric
import argparse
import os

def load_csv(file):
    dataset = load_dataset('csv', data_files=file)
    dataset = dataset['train']
    dataset = dataset.train_test_split(test_size=0.1)
    train_set = dataset['train']
    dev_set = dataset['test']
    return train_set, dev_set

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

    def __init__(self, args):
        self.args = args
        self.model = self.args.model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenized_train_set, self.tokenized_dev_set = tokenize_data(load_csv(self.args.train_set), self.tokenizer)
        self.tokenized_test_set = tokenize_data(load_csv(self.args.test_set), self.tokenizer) if args.test_set else None
        #set training args
        self.training_args = self.get_training_args()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=2)
        #initialize trainer
        self.trainer = Trainer(
          model=self.model,
          args=self.training_args,
          train_dataset=self.tokenized_train_set,
          eval_dataset=self.tokenized_dev_set,
          compute_metrics=compute_metrics
        )

    def get_training_args(self):

        training_args = TrainingArguments(evaluation_strategy="epoch", do_train=True, output_dir=self.args.output_dir)
        training_args.num_train_epochs = self.args.num_train_epochs
        training_args.save_steps = self.args.save_steps
        training_args.per_device_train_batch_size = self.args.per_device_train_batch_size
        training_args.per_device_eval_batch_size = self.args.per_device_eval_batch_size
        training_args.overwrite_output_dir = self.args.overwrite_output_dir
        return training_args

    def write_predictions(self, predictions):
        _, predicted = torch.max(torch.from_numpy(predictions), 1)
        output_predict_file = os.path.join(self.args.output_dir, f"predict_results.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predicted):
                writer.write(f"{index}\t{item}\n")

    def train_model(self):
        #train model
        self.trainer.train()

        if self.args.save_metrics:
            # compute evaluation results
            eval_metrics = self.trainer.evaluate()
            eval_metrics["eval_samples"] = len(self.tokenized_dev_set)
            # save evaluation results
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)

        if self.tokenized_test_set:
            predictions = self.trainer.predict(self.tokenized_test_set)
            self.write_predictions(predictions.predictions)
            if self.args.save_metrics:
                self.trainer.log_metrics("predict", predictions.metrics)
                self.trainer.save_metrics("predict", predictions.metrics)

        if self.args.save_model:
            self.model.save_pretrained("model")

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
                        help = ".csv dataset to used for prediction",
                        required = False,
                        default = None,
                        type = str)
    parser.add_argument("-num_train_epochs",
                        help = "number of epochs for training",
                        required = False,
                        default=3.0,
                        type = float)
    parser.add_argument("-save_steps",
                        help = "number of updates steps before two checkpoint saves",
                        required = False,
                        default=50000,
                        type = int)
    parser.add_argument("-per_device_train_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for training",
                        required = False,
                        default=32,
                        type = int)
    parser.add_argument("-per_device_eval_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for evaluation",
                        required = False,
                        default=32,
                        type = int)
    parser.add_argument("-output_dir",
                        help = "output directory",
                        required = False,
                        default='outs',
                        type = str)
    parser.add_argument("-overwrite_output_dir",
                        help = "overwrite previous output dir",
                        required = False,
                        default=True,
                        type = bool)
    parser.add_argument("-save_metrics",
                        help = "set to true to save metrics",
                        required = False,
                        default=False,
                        type = bool)
    parser.add_argument("-save_model",
                        help = "set to true to save model",
                        required = False,
                        default=False,
                        type = bool)
    args = parser.parse_args()

    modelTrainer = ModelTrainer(args)
    modelTrainer.train_model()

if __name__=='__main__':
    main()
