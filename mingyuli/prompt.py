import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

try:
    os.environ['DATASETS_CACHE'] = '/datasets'
except:
    print('failed to download datasets to projects')
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets
import tqdm
from transformers import get_scheduler

import pandas as pd

from sklearn.model_selection import train_test_split

import yacs
import tensorboardX

import argparse

#import wandb

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

def get_parser(
    parser = argparse.ArgumentParser(
        description="prompt tuning a language model for Task1")
    ):
    parser.add_argument('--model', type=str, default="bert-base-uncased")
    parser.add_argument('--device',type =torch.device, 
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--save_dir',
                        type=pathlib.Path,
                        default=pathlib.Path.cwd() / 'models')
    return parser


def construct_datasets(args,tokenizer,WrapperClass):
    ## Dataset1: IMDB
    data1 = pd.read_csv('datasets/news_d.csv')
    #data1 = pd.read_excel('datasets/USPVD2010.xlsx')
    dataset_data1 = Dataset.from_pandas(data1)

    ## Dataset2: FCE
    #data2 = pd.read_csv('datasets/CTB_forCASE_rsampled.csv')
    #data2 = pd.read_csv('datasets/val_10k.csv')
    #dataset_data2 = Dataset.from_pandas(data2)

    processed_data1 = [InputExample(label = int(n['category']),text_a = (n['headline'])) for n in dataset_data1]
    #processed_data2 = [InputExample(label = int(n['label']),text_a = n['text']) for n in dataset_data2]

    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} It is {"mask"}.',
        tokenizer=tokenizer,
    )

    #classes = ['1','2','3','4','5']
    classes = ['0','1','2','3','4','5','6']


    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words = {
            "0": ["0"],
            "1": ["1"],
            "2": ["2"],
            "3": ["3"],
            "4": ["4"],
            "5": ["5"],
            "6": ["6"]
        },
        tokenizer=tokenizer,
    )
#     processed_total = processed_imdb + processed_fce
    processed_total = processed_data1 #+processed_data2
    processed_train, processed_dev = train_test_split(
        processed_total,
        test_size=0.2,
        random_state=42
    )
    train_loader = PromptDataLoader(
        dataset=processed_train,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size,
    )
    dev_loader = PromptDataLoader(
        dataset=processed_dev,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size,
    )
    return train_loader,dev_loader,promptVerbalizer,promptTemplate

def train(args):
    plm, tokenizer, model_config, WrapperClass = load_plm("bert","bert-base-uncased")

    train_loader,dev_loader,promptVerbalizer,promptTemplate = construct_datasets(
        args,
        tokenizer,
        WrapperClass
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    optimizer = optim.AdamW(
        promptModel.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    metric1 = datasets.load_metric("accuracy")
    promptModel = promptModel.to(args.device)
    torch.save(promptModel,args.save_dir/"model.pt")
    
    #wandb.init(project="CASE")
    #wandb.watch(promptModel)
    #print(torch.cuda.memory_allocated(device=None)/1024/1024/1024)
    
    for epoch in tqdm.trange(args.epochs, desc="Epochs"):
        tot_loss = 0
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}",
            total=len(train_loader),
            disable=None,
            leave=False,
        )
        promptModel.train()
        for step, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            logits = promptModel(batch)
            # print(logits.dtype)
            preds = torch.argmax(logits, dim=-1)
            labels = batch['label']
            # print('preds',preds,preds.dtype)
            # print('labels',labels,labels.dtype)
            loss = criterion(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            metric1.add_batch(predictions=preds, references=labels)
            # metric2.add_batch(predictions=preds,references = labels,average='micro')
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
                accuracy = metric1.compute()
                print(accuracy)
                #wandb.log({'loss':tot_loss,'accuracy':accuracy})

        pbar.close()
        torch.save(promptModel,args.save_dir/"model.pt")
        print('model saved')
        
    #train_loader.save(args.save_dir / "train_dataset.pt")
    #dev_loader.save(args.save_dir / "dev_dataset.pt")
    try:
        torch.save(train_loader,args.save_dir/"train.pt")
        torch.save(dev_loader,args.save_dir/"dev.pt")
    except:
        print('cannot save the two datasets.')
def main(args):
    print('starting training')
    train(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)