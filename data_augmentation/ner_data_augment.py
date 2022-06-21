import argparse
from csv import reader, writer
from transformers import pipeline
from datasets import load_dataset
from copy import deepcopy
import spacy

nlp = spacy.load("en_core_web_sm")

def augment(infile, outfile):
    with open(outfile, 'w') as out:
        dataset = load_dataset('csv', data_files=infile)
        dataset = dataset['train']
        dataset = dataset.remove_columns([feature for feature in dataset.features if feature != 'text' and feature != 'label'])

        fillmask = pipeline('fill-mask', model='roberta-base')
        mask_token = fillmask.tokenizer.mask_token

        def augment(sentence):
            doc = nlp(sentence)
            to_replace = [str(ent) for ent in doc.ents]
            new_sents = [sentence]
            if to_replace:
                def get_augmented_sentences(sentence, to_replace):
                    to_return = []
                    for i in range(3):
                        def fill_mask(word):
                            masked_sentence = sentence.replace(word, mask_token, 1)
                            changed_word = fillmask(masked_sentence)[i]['token_str']
                            return changed_word
                        changed_words = list(map(fill_mask, to_replace))
                        alter_sentence = deepcopy(sentence)
                        for i, word in enumerate(changed_words):
                            alter_sentence = alter_sentence.replace(' ' + to_replace[i], word)
                        to_return.append(alter_sentence)
                    return to_return
                new_sents.extend(get_augmented_sentences(sentence, to_replace))
            return {'data': new_sents}

        aug_dataset = dataset.map(lambda examples: augment(examples['text']))

        csv_writer = writer(out)
        csv_writer.writerow(['text','label'])
        for row in aug_dataset:
            for sentence in row['data']:
                csv_writer.writerow([sentence, row['label']])

def main():
    parser = argparse.ArgumentParser(description = "augment data using NER mask filling")
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
