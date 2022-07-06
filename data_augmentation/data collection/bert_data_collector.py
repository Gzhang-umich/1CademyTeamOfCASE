import argparse
from csv import reader, writer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from scipy.special import softmax
import numpy as np

def collect(infile, outfile):
    with open(outfile, 'w') as out:
        dataset = load_dataset('csv', data_files=infile)
        dataset = dataset['train']
        dataset = dataset.remove_columns([feature for feature in dataset.features if feature != 'text'])

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        model = AutoModelForSequenceClassification.from_pretrained("model/")

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        trainer = Trainer(model=model)

        predictions = trainer.predict(tokenized_dataset)

        csv_writer = writer(out)
        csv_writer.writerow(['text','label'])
        for text, logit in zip(tokenized_dataset['text'], predictions.predictions):
            if max(softmax(logit)) > 0.9:
                csv_writer.writerow([text, np.argmax(logit)])

def main():
    parser = argparse.ArgumentParser(description = "data collector")
    parser.add_argument("-datafile",
                        help = "name or path to .csv data",
                        required = True,
                        type = str)
    parser.add_argument("-outfile",
                        help = ".csv of augmented data",
                        required = True,
                        type = str)
    args = parser.parse_args()
    collect(args.datafile, args.outfile)

if __name__ == '__main__':
    main()
