#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import re
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import datasets
from sklearn.model_selection import train_test_split
import torch
from datasets import ClassLabel, load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy 

from eval_st2 import main as evaluate
from model_st2 import ST2Model
# from transformers.utils import get_full_repo_name, send_example_telemetry
# from transformers.utils.versions import require_version


logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def merge_multi_spaces(sent):
    return re.sub(' +',' ',sent)


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub('</*[A-Z]+\d*>','',tok) 


def get_BIO(text_w_pairs):
    tokens = []
    ce_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):

        # Replace if special
        if '<ARG0>' in tok:
            tok = re.sub('<ARG0>','',tok)
            tag = 'B-C'
            next_tag = 'I-C'
        elif '</ARG0>' in tok:
            tok = re.sub('</ARG0>','',tok)
            tag = 'I-C'
            next_tag = 'O'
        elif '<ARG1>' in tok:
            tok = re.sub('<ARG1>','',tok)
            tag = 'B-E'
            next_tag = 'I-E'
        elif '</ARG1>' in tok:
            tok = re.sub('</ARG1>','',tok)
            tag = 'I-E'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        ce_tags.append(tag)
        tag = next_tag
    
    return tokens, ce_tags


def get_BIO_sig(text_w_pairs):
    tokens = []
    s_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):
        # Replace if special
        if '<SIG' in tok:
            tok = re.sub('<SIG([A-Z]|\d)*>','',tok)
            tag = 'B-S'
            next_tag = 'I-S'
            if '</SIG' in tok: # one word only
                tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
                next_tag = 'O'

        elif '</SIG' in tok:
            tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
            tag = 'I-S'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        s_tags.append(tag)
        tag = next_tag
    
    return tokens, s_tags


def get_BIO_all(text_w_pairs):
    tokens, ce_tags = get_BIO(text_w_pairs)
    tokens_s, s_tags = get_BIO_sig(text_w_pairs)
    assert(tokens==tokens_s)
    assert(len(ce_tags)==len(s_tags)==len(tokens))
    return tokens, ce_tags, s_tags


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_ner_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if "validation" not in data_files:
            raw_datasets = load_dataset(extension, data_files=data_files)['train'].train_test_split(300, shuffle=False)
            raw_datasets = {'train': raw_datasets['train'], 'validation': raw_datasets['test']}
            raw_datasets = DatasetDict(raw_datasets)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files)['train'].train_test_split()
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # if raw_datasets["train"] is not None:
    #     column_names = raw_datasets["train"].column_names
    #     features = raw_datasets["train"].features
    # else:
    #     column_names = raw_datasets["validation"].column_names
    #     features = raw_datasets["validation"].features

    # if args.text_column_name is not None:
    #     text_column_name = args.text_column_name
    # elif "tokens" in column_names:
    #     text_column_name = "tokens"
    # else:
    #     text_column_name = column_names[0]

    # if args.label_column_name is not None:
    #     label_column_name = args.label_column_name
    # elif f"{args.task_name}_tags" in column_names:
    #     label_column_name = f"{args.task_name}_tags"
    # else:
    #     label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    # def get_label_list(labels):
    #     unique_labels = set()
    #     for label in labels:
    #         unique_labels = unique_labels | set(label)
    #     label_list = list(unique_labels)
    #     label_list.sort()
    #     return label_list

    # # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # # Otherwise, we have to get the list of labels manually.
    # labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    # if labels_are_int:
    #     label_list = features[label_column_name].feature.names
    #     label_to_id = {i: i for i in range(len(label_list))}
    # else:
    #     label_list = get_label_list(raw_datasets["train"][label_column_name])
    #     label_to_id = {l: i for i, l in enumerate(label_list)}

    ce_label_list = ['O', 'B-C', 'I-C', 'B-E', 'I-E']
    ce_label_to_id = {l: i for i, l in enumerate(ce_label_list)}
    ce_id_to_label = {i: l for i, l in enumerate(ce_label_list)}
    
    sig_label_list = ['O', 'B-S', 'I-S']
    sig_label_to_id = {l: i for i, l in enumerate(sig_label_list)}
    sig_id_to_label = {i: l for i, l in enumerate(sig_label_list)}

    # num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<ARG0>", "</ARG0>", "<ARG1>", "</ARG1>", "<SIG0>", "</SIG0>"]})

    model = ST2Model(args)

    # if args.model_name_or_path:
    #     model = AutoModelForTokenClassification.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForTokenClassification.from_config(config)

    # model.resize_token_embeddings(len(tokenizer))


    # # Model has labels -> use them.
    # if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    #     if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
    #         # Reorganize `label_list` to match the ordering of the model.
    #         if labels_are_int:
    #             label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #         else:
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #             label_to_id = {l: i for i, l in enumerate(label_list)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
    #             f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
    #         )

    # Set the correspondences label/ID inside the model config
    # model.config.label2id = {l: i for i, l in enumerate(label_list)}
    # model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    ce_b_to_i_label = []
    for idx, label in enumerate(ce_label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in ce_label_list:
            ce_b_to_i_label.append(ce_label_list.index(label.replace("B-", "I-")))
        else:
            ce_b_to_i_label.append(idx)

    sig_b_to_i_label = []
    for idx, label in enumerate(sig_label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in sig_label_list:
            sig_b_to_i_label.append(sig_label_list.index(label.replace("B-", "I-")))
        else:
            sig_b_to_i_label.append(idx)

    def preprocessing(examples):
        all_tokens = []
        all_ce_tags = []
        all_s_tags = []
        outputs = {}
        
        for k in examples.keys():
            outputs[k] = []

        for i, causal_text_w_pairs in enumerate(examples["causal_text_w_pairs"]):
            causal_text_w_pairs = eval(causal_text_w_pairs)
            if len(causal_text_w_pairs) > 0:
                for text in causal_text_w_pairs:
                    tokens, ce_tags, s_tags = get_BIO_all(text)
                    all_tokens.append(tokens)
                    all_ce_tags.append(ce_tags)
                    all_s_tags.append(s_tags)
                    for k, v in examples.items():
                        outputs[k].append(v[i])
            # else:
            #     tokens, ce_tags, s_tags = get_BIO_all(examples["text"][i])
            #     all_tokens.append(tokens)
            #     all_ce_tags.append(ce_tags)
            #     all_s_tags.append(s_tags)       
            #     for k, v in examples.items():
            #         outputs[k].append(v[i])
        return {"tokens": all_tokens, "ce_tags": all_ce_tags, "s_tags": all_s_tags, **outputs}

    raw_datasets['train'] = raw_datasets['train'].map(preprocessing, batched=True, remove_columns=raw_datasets['train'].column_names)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        ce_labels = []
        for i, label in enumerate(examples['ce_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(ce_label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(ce_b_to_i_label[ce_label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            ce_labels.append(label_ids)

        sig_labels = []
        for i, label in enumerate(examples['s_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(sig_label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(sig_b_to_i_label[sig_label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            sig_labels.append(label_ids)

        tokenized_inputs["ce_labels"] = ce_labels
        tokenized_inputs["sig_labels"] = sig_labels
        return tokenized_inputs

    def tokenize(examples):
        tokenized_inputs = tokenizer(
            [text.split() for text in examples["text"]],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        tokenized_inputs["text"] = examples["text"]
        tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(examples["text"]))]
        return tokenized_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets['train'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    # train_dataset = processed_raw_datasets["train"]
    # eval_dataset = processed_raw_datasets["validation"]
    with accelerator.main_process_first():
        eval_dataset = raw_datasets["validation"].map(
            tokenize,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on dataset",         
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        @dataclass
        class DataCollatorForTokenClassification(DataCollatorMixin):
            """
            Data collator that will dynamically pad the inputs received, as well as the labels.

            Args:
                tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
                    The tokenizer used for encoding the data.
                padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                    Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                    among:

                    - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
                    is provided).
                    - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                    acceptable input length for the model if that argument is not provided.
                    - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                    lengths).
                max_length (`int`, *optional*):
                    Maximum length of the returned list and optionally padding length (see above).
                pad_to_multiple_of (`int`, *optional*):
                    If set will pad the sequence to a multiple of the provided value.

                    This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                    7.5 (Volta).
                label_pad_token_id (`int`, *optional*, defaults to -100):
                    The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
                return_tensors (`str`):
                    The type of Tensor to return. Allowable values are "np", "pt" and "tf".
            """

            tokenizer: PreTrainedTokenizerBase
            padding: Union[bool, str, PaddingStrategy] = True
            max_length: Optional[int] = None
            pad_to_multiple_of: Optional[int] = None
            label_pad_token_id: int = -100
            return_tensors: str = "pt"

            def torch_call(self, features):
                if "ce_labels" in features[0]:
                    ce_labels = [feature["ce_labels"] for feature in features]
                    sig_labels = [feature["sig_labels"] for feature in features]

                word_ids = None
                if "word_ids" in features[0]:
                    word_ids = [features[i].pop("word_ids") for i, feature in enumerate(features)]
                    texts = [features[i].pop("text") for i, feature in enumerate(features)]

                batch = self.tokenizer.pad(
                    features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    # Conversion to tensors will fail if we have labels as they are not of the same length yet.
                    return_tensors=None,
                )

                sequence_length = torch.tensor(batch["input_ids"]).shape[1]
                assert self.tokenizer.padding_side == "right"

                if "ce_labels" in features[0]:
                    batch["ce_labels"] = [
                        list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in ce_labels
                    ]
                    batch["sig_labels"] = [
                        list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in sig_labels
                    ]

                batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
                if word_ids is not None:
                    batch["word_ids"] = word_ids
                    batch["text"] = texts
                return batch
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("ner_no_trainer", experiment_config)

    # Metrics
    # metric = load_metric("seqeval")

    # def get_labels(predictions, references):
    #     # Transform predictions and references tensos to numpy arrays
    #     if device.type == "cpu":
    #         y_pred = predictions.detach().clone().numpy()
    #         y_true = references.detach().clone().numpy()
    #     else:
    #         y_pred = predictions.detach().cpu().clone().numpy()
    #         y_true = references.detach().cpu().clone().numpy()

    #     # Remove ignored index (special tokens)
    #     true_predictions = [
    #         [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
    #         for pred, gold_label in zip(y_pred, y_true)
    #     ]
    #     true_labels = [
    #         [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
    #         for pred, gold_label in zip(y_pred, y_true)
    #     ]
    #     return true_predictions, true_labels

    # def compute_metrics():
        # evaluate(truth, predictions, calculate_best_combi=True)
        # results = metric.compute()
        # if args.return_entity_level_metrics:
        #     # Unpack nested dictionaries
        #     final_results = {}
        #     for key, value in results.items():
        #         if isinstance(value, dict):
        #             for n, v in value.items():
        #                 final_results[f"{key}_{n}"] = v
        #         else:
        #             final_results[key] = value
        #     return final_results
        # else:
        #     return {
        #         "precision": results["overall_precision"],
        #         "recall": results["overall_recall"],
        #         "f1": results["overall_f1"],
        #         "accuracy": results["overall_accuracy"],
        #     }

    # Train!
    truth = pd.read_csv(args.train_file)[-300:].reset_index(drop=True)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs["loss"]
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss}")
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # if epoch % 5 == 0:
        model.eval()
        samples_seen = 0

        predictions = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**{k: batch[k] if k in batch else None for k in ["input_ids", "attention_mask", "token_type_ids"]})
            ce_predictions = outputs["ce_logits"].argmax(dim=-1).tolist()
            sig_predictions = outputs["sig_logits"].argmax(dim=-1).tolist()
            
            for i in range(len(ce_predictions)):
                word_ids = batch["word_ids"][i]
                space_splitted_tokens = batch["text"][i].split()
                tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])[:len(word_ids)]
                for j, (token, id) in enumerate(zip(tokens, word_ids)):
                    if j == 0 or j == len(tokens) - 1:
                        continue

                    if ce_id_to_label[ce_predictions[i][j]] == 'B-C':
                        space_splitted_tokens[id] = '<ARG0>' + space_splitted_tokens[id]
                    elif ce_id_to_label[ce_predictions[i][j]] == 'I-C' and (j == len(tokens) - 1 or ce_id_to_label[ce_predictions[i][j + 1]]) == 'O':
                        space_splitted_tokens[id] += '</ARG0>'
                    elif ce_id_to_label[ce_predictions[i][j]] == 'B-E':
                        space_splitted_tokens[id] = '<ARG1>' + space_splitted_tokens[id] 
                    elif ce_id_to_label[ce_predictions[i][j]] == 'I-E' and (j == len(tokens) - 1 or ce_id_to_label[ce_predictions[i][j + 1]] == 'O'):
                        space_splitted_tokens[id] += '</ARG1>'
                    
                    if sig_id_to_label[sig_predictions[i][j]] == 'B-S':
                        space_splitted_tokens[id] = '<SIG0>' + space_splitted_tokens[id]
                    elif sig_id_to_label[sig_predictions[i][j]] == 'I-S' and (j == len(tokens) - 1 or sig_id_to_label[sig_predictions[i][j + 1]] == 'O'):
                        space_splitted_tokens[id] += '</SIG0>'

                predictions.append([' '.join(space_splitted_tokens)])

        main_results = evaluate(truth, predictions)

        logger.info("Cause | P: {} | R: {} | F1: {}".format(main_results["Cause"]["precision"], main_results["Cause"]["recall"], main_results["Cause"]["f1"]))
        logger.info("Effect | P: {} | R: {} | F1: {}".format(main_results["Effect"]["precision"], main_results["Effect"]["recall"], main_results["Effect"]["f1"]))
        logger.info("Signal | P: {} | R: {} | F1: {}".format(main_results["Signal"]["precision"], main_results["Signal"]["recall"], main_results["Signal"]["f1"]))
        logger.info("Overall | P: {} | R: {} | F1: {}".format(main_results["Overall"]["precision"], main_results["Overall"]["recall"], main_results["Overall"]["f1"]))

            
        #     labels = batch["labels"]
        #     if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
        #         predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        #         labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        #     predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
        #     # If we are in a multiprocess environment, the last batch has duplicates
        #     if accelerator.num_processes > 1:
        #         if step == len(eval_dataloader) - 1:
        #             predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
        #             labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
        #         else:
        #             samples_seen += labels_gathered.shape[0]
        #     # preds, refs = get_labels(predictions_gathered, labels_gathered)
        #     # metric.add_batch(
        #     #     predictions=preds,
        #     #     references=refs,
        #     # )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        # eval_metric = compute_metrics()
        # accelerator.print(f"epoch {epoch}:", eval_metric)
        # if args.with_tracking:
        #     accelerator.log(
        #         {
        #             "seqeval": eval_metric,
        #             "train_loss": total_loss.item() / len(train_dataloader),
        #             "epoch": epoch,
        #             "step": completed_steps,
        #         },
        #         step=completed_steps,
        #     )

        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
        #         repo.push_to_hub(
        #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
        #         )

        # if args.checkpointing_steps == "epoch":
        #     output_dir = f"epoch_{epoch}"
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)
        #     accelerator.save_state(output_dir)

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #     )
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    #     with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    #         json.dump(
    #             {"eval_accuracy": eval_metric["accuracy"], "train_loss": total_loss.item() / len(train_dataloader)}, f
    #         )


if __name__ == "__main__":
    main()