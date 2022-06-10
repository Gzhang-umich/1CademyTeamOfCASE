import argparse
from csv import writer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

model_pairs = [("facebook/wmt19-en-de", "facebook/wmt19-de-en"),
             ("facebook/wmt19-en-ru", "facebook/wmt19-ru-en")]

def augment(infile, outfile):
    with open(outfile, 'w') as out:
        dataset = load_dataset('csv', data_files=infile)
        dataset = dataset['train']
        dataset = dataset.remove_columns([feature for feature in dataset.features if feature != 'text' and feature != 'label'])

        csv_writer = writer(out)
        csv_writer.writerow(['text','label'])

        for row in dataset:
            csv_writer.writerow([row['text'], row['label']])

        def encode(examples, tokenizer):
            return tokenizer(examples['text'], truncation=True, padding=True)

        def make_output(examples, model):
            output = model.generate(input_ids=torch.as_tensor(examples['input_ids']))
            output_ids = []
            for example in output:
                output_ids.append(example)
            return {'output_ids': output_ids}

        def translate(examples, tokenizer):
            text = []
            for example in examples['output_ids']:
              translated = tokenizer.decode(example, skip_special_tokens=True)
              text.append(translated)
            return {'text':text}

        for eng, foreign in model_pairs:
            eng_tokenizer = AutoTokenizer.from_pretrained(eng)
            eng_model = AutoModelForSeq2SeqLM.from_pretrained(eng)
            foreign_tokenizer = AutoTokenizer.from_pretrained(foreign)
            foreign_model = AutoModelForSeq2SeqLM.from_pretrained(foreign)

            encoded = dataset.map(lambda examples: encode(examples, eng_tokenizer), batched=True, batch_size=8)
            foreign_ids = encoded.map(lambda examples: make_output(examples, eng_model), batched=True, batch_size=8)
            translated = foreign_ids.map(lambda examples: translate(examples, eng_tokenizer), batched=True, batch_size=8)

            foreign_encoded = translated.map(lambda examples: encode(examples, foreign_tokenizer), batched=True, batch_size=8)
            eng_ids = foreign_encoded.map(lambda examples: make_output(examples, foreign_model), batched=True, batch_size=8)
            eng_translated = eng_ids.map(lambda examples: translate(examples, foreign_tokenizer), batched=True, batch_size=8)

            for row in eng_translated:
                csv_writer.writerow([row['text'], row['label']])

def main():
    parser = argparse.ArgumentParser(description = "augment data using Seq2Seq Modelling")
    parser.add_argument("-datafile",
                        help = "name or path to .csv data",
                        required = True,
                        type = str)
    parser.add_argument("-outfile",
                        help = ".csv of augmented data",
                        required = True,
                        type = str)
    args = parser.parse_args()
    augment(args.datafile, args.outfile)

if __name__ == '__main__':
    main()
