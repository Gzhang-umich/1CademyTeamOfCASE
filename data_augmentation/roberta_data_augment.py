import argparse
from csv import reader, writer
from random import sample, randint
from transformers import pipeline
from datasets import load_dataset

def augment(infile, outfile):
    with open(outfile, 'w') as out:
        dataset = load_dataset('csv', data_files=infile)
        dataset = dataset['train']
        dataset = dataset.remove_columns([feature for feature in dataset.features if feature != 'text' and feature != 'label'])

        fillmask = pipeline('fill-mask', model='roberta-base')
        mask_token = fillmask.tokenizer.mask_token

        def augment_data(examples):
            outputs = []
            for sentence in examples['text']:
                words = sentence.split(' ')
                K = randint(1, len(words)-1)
                masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
                predictions = fillmask(masked_sentence)
                augmented_sequences = [predictions[i]['sequence'] for i in range(3)]
                outputs += [sentence] + augmented_sequences
            return {'data': outputs}

        aug_dataset = dataset.map(augment_data, batched=True, remove_columns=dataset.column_names, batch_size=8)

        csv_writer = writer(out)
        csv_writer.writerow(['text','label'])
        for i, text in enumerate(aug_dataset['data']):
            csv_writer.writerow([text, dataset['label'][i // 4]])

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
