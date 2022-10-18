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
import copy
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from datetime import datetime

import datasets
from sklearn.model_selection import train_test_split
import torch
from datasets import ClassLabel, load_dataset, load_metric, DatasetDict, concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
# from accelerate.utils import set_seed
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
from model_st2 import ST2Model, ST2ModelV2, SignalDetector

import wandb
import numpy as np
from sklearn.metrics import accuracy_score
# from transformers.utils import get_full_repo_name, send_example_telemetry
# from transformers.utils.versions import require_version


logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


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
        default=256,
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
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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

    # our arguments
    parser.add_argument(
        "--add_signal_bias",
        action="store_true",
        help="Whether or not to add signal bias",
    )

    parser.add_argument(
        "--signal_bias_on_top_of_lm",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    
    parser.add_argument(
        "--postprocessing_position_selector",
        action="store_true",
        help="Whether or not to use postprocessing position selector to control overlap problem.",
    )

    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Whether or not to add MLP layer on top of the pretrained LM.",
    )

    parser.add_argument(
        "--signal_classification",
        action="store_true",
        help="Conduct signal classification to verify whether we need to detect signal span.",
    )

    parser.add_argument(
        "--pretrained_signal_detector",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )

    parser.add_argument(
        "--tapt",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )

    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="classifier dropout rate",
    )

    parser.add_argument(
        "--load_checkpoint_for_test",
        type=str,
        default=None,
        help="classifier dropout rate",
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to use pretrained signal detector",
    ) 

    parser.add_argument(
        "--augmentation_file",
        type=str,
        default=None,
        help="Whether to use pretrained signal detector",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Whether to use pretrained signal detector",
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
    
    wandb.init()
    wandb.run.log_code(".")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.signal_classification and args.pretrained_signal_detector:
        signal_detector = SignalDetector()

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
            raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    if args.augmentation_file is not None:
        augment_dataset = Dataset.from_pandas(pd.read_csv(args.augmentation_file))

    # raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], augment_dataset])
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

    model = ST2ModelV2(args)

    if args.tapt:
        model_checkpoint = dict()
        for k, v in torch.load("/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/mlm/pytorch_model.bin").items():
            model_checkpoint[k.replace('albert.', '')] = v
        model.model.load_state_dict(model_checkpoint, strict=False)
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
        return {"tokens": all_tokens, "ce_tags": all_ce_tags, "s_tags": all_s_tags}

    raw_datasets['train'] = raw_datasets['train'].map(preprocessing, batched=True, remove_columns=raw_datasets['train'].column_names)

    if args.augmentation_file is not None:
        augment_dataset = augment_dataset.map(preprocessing, batched=True, remove_columns=augment_dataset.column_names)
        raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], augment_dataset])

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    if args.add_signal_bias:
        # build signal phrases lexicon to add bias to the model
        signal_phrases = []
        with open('/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/signal_phrases.txt', 'r') as f:
            for line in f.readlines():
                signal_phrases.append(line.strip())
        
        signal_phrases_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) for s in signal_phrases]

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
        
        start_positions = []
        end_positions = []
        # ce_labels = []
        for i, label in enumerate(examples['ce_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            # label_ids = []

            arg1_start_position = -100
            arg1_end_position = -100
            arg0_start_position = -100
            arg0_end_position = -100
            for j, word_idx in enumerate(word_ids):
                if word_idx is not None and label[word_idx] == 'B-E':  # and (previous_word_idx is None or label[previous_word_idx] in ['O'):
                    arg1_start_position = j
                if word_idx is not None and label[word_idx] in ['O', 'B-C'] and (previous_word_idx is not None and label[previous_word_idx] in ['I-E', 'B-E']):
                    arg1_end_position = j - 1

                if word_idx is not None and label[word_idx] == 'B-C':  # and (previous_word_idx is None or label[previous_word_idx] == 'O'):
                    arg0_start_position = j
                if word_idx is not None and label[word_idx] in ['O', 'B-E'] and (previous_word_idx is not None and label[previous_word_idx] in ['I-C', 'B-C']):
                    arg0_end_position = j - 1

                if j == len(word_ids) - 1:
                    if previous_word_idx is not None and label[previous_word_idx] in ['I-C', 'B-C']:
                        arg0_end_position = j - 1

                    if previous_word_idx is not None and label[previous_word_idx] in ['I-E', 'B-E']:
                        arg1_end_position = j - 1
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                # if word_idx is None:
                #     label_ids.append(-100)
                # # We set the label for the first token of each word.
                # elif word_idx != previous_word_idx:
                #     label_ids.append(ce_label_to_id[label[word_idx]])
                # # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # # the label_all_tokens flag.
                # else:
                #     if args.label_all_tokens:
                #         label_ids.append(ce_b_to_i_label[ce_label_to_id[label[word_idx]]])
                #     else:
                #         label_ids.append(-100)
                previous_word_idx = word_idx
            
            assert arg1_start_position != -100 and arg1_end_position != -100
            assert arg0_start_position != -100 and arg0_end_position != -100

            start_positions.append([arg0_start_position, arg1_start_position])
            end_positions.append([arg0_end_position, arg1_end_position])
            # ce_labels.append(label_ids)

        # sig_labels = []
        for i, label in enumerate(examples['s_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            # label_ids = []

            sig_start_position = -100
            sig_end_position = -100
            for j, word_idx in enumerate(word_ids):
                if word_idx is not None and label[word_idx] == 'B-S' and (previous_word_idx is None or label[previous_word_idx] == 'O'):
                    sig_start_position = j
                if word_idx is not None and label[word_idx] == 'O' and (previous_word_idx is not None and label[previous_word_idx] in ['I-S', 'B-S']):
                    sig_end_position = j - 1

                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                # if word_idx is None:
                #     label_ids.append(-100)
                # # We set the label for the first token of each word.
                # elif word_idx != previous_word_idx:
                #     label_ids.append(sig_label_to_id[label[word_idx]])
                # # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # # the label_all_tokens flag.
                # else:
                #     if args.label_all_tokens:
                #         label_ids.append(sig_b_to_i_label[sig_label_to_id[label[word_idx]]])
                #     else:
                #         label_ids.append(-100)
                previous_word_idx = word_idx
            start_positions[i].append(sig_start_position)
            end_positions[i].append(sig_end_position)
            # sig_labels.append(label_ids)
        
        if args.add_signal_bias:
            all_signal_bias_mask = []
            for i, input_id in enumerate(tokenized_inputs["input_ids"]):
                signal_bias_mask = [0] * len(input_id)
                for j, phrase_id in enumerate(signal_phrases_ids):
                    for k in range(len(input_id)):
                        if input_id[k:k+len(phrase_id)] == phrase_id:
                            signal_bias_mask[k:k+len(phrase_id)] = [1] * len(phrase_id)
                all_signal_bias_mask.append(signal_bias_mask)
            tokenized_inputs["signal_bias_mask"] = all_signal_bias_mask
            

        # tokenized_inputs["ce_labels"] = ce_labels
        # tokenized_inputs["sig_labels"] = sig_labels
        tokenized_inputs["start_positions"] = start_positions
        tokenized_inputs["end_positions"] = end_positions
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

        if args.add_signal_bias:
            all_signal_bias_mask = []
            for i, input_id in enumerate(tokenized_inputs["input_ids"]):
                signal_bias_mask = [0] * len(input_id)
                for j, phrase_id in enumerate(signal_phrases_ids):
                    for k in range(len(input_id)):
                        if input_id[k:k+len(phrase_id)] == phrase_id:
                            signal_bias_mask[k:k+len(phrase_id)] = [1] * len(phrase_id)
                all_signal_bias_mask.append(signal_bias_mask)
            tokenized_inputs["signal_bias_mask"] = all_signal_bias_mask

        tokenized_inputs["text"] = examples["text"]
        tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(i) for i in range(len(examples["text"]))]
        return tokenized_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets['train'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False,
        )

    # train_dataset = processed_raw_datasets["train"]
    # eval_dataset = processed_raw_datasets["validation"]
    with accelerator.main_process_first():
        eval_dataset = raw_datasets["validation"].map(
            tokenize,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False,      
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
                if "start_positions" in features[0]:
                    # ce_labels = [feature["ce_labels"] for feature in features]
                    # sig_labels = [feature["sig_labels"] for feature in features]
                    start_positions = [feature["start_positions"] for feature in features]
                    end_positions = [feature["end_positions"] for feature in features]


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

                if "start_positions" in features[0]:
                    batch["start_positions"] = start_positions
                    batch["end_positions"] = end_positions

                # pad signal_bias_mask to max_length
                if args.add_signal_bias:
                    for i, signal_bias_mask in enumerate(batch['signal_bias_mask']):
                        batch['signal_bias_mask'][i] = signal_bias_mask + [0] * (sequence_length - len(signal_bias_mask))

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

    if not args.do_test:
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

        # Train!
        if args.validation_file is None:
            truth = pd.read_csv(args.train_file, sep=",", encoding="utf-8")[-300:].reset_index(drop=True)
        else:
            truth = pd.read_csv(args.validation_file, sep=",", encoding="utf-8").reset_index(drop=True)
        
        best_overall_f1 = 0.
        best_cause_f1 = 0.
        best_effect_f1 = 0.
        best_signal_f1 = 0.
        
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

            model.eval()

            predictions = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**{k: batch[k] if k in batch else None for k in ["input_ids", "attention_mask", "token_type_ids", "signal_bias_mask"]})
                # ce_predictions = outputs["ce_logits"].argmax(dim=-1).tolist()
                # sig_predictions = outputs["sig_logits"].argmax(dim=-1).tolist()
                start_cause_predictions = outputs["start_arg0_logits"]
                end_cause_predictions = outputs["end_arg0_logits"]

                start_effect_predictions = outputs["start_arg1_logits"]
                end_effect_predictions = outputs["end_arg1_logits"]

                start_signal_predictions = outputs["start_sig_logits"]
                end_signal_predictions = outputs["end_sig_logits"]

                for i in range(len(batch["input_ids"])):
                    word_ids = batch["word_ids"][i]
                    space_splitted_tokens = batch["text"][i].split(" ")

                    if args.postprocessing_position_selector:
                        if not args.beam_search:
                            start_cause, end_cause, start_effect, end_effect = model.position_selector(
                                start_cause_logits=start_cause_predictions[i],
                                end_cause_logits=end_cause_predictions[i],
                                start_effect_logits=start_effect_predictions[i],
                                end_effect_logits=end_effect_predictions[i],
                                attention_mask=batch["attention_mask"][i],
                                word_ids=word_ids,
                            )
                        else:
                            indices1, indices2, score1, score2, topk_scores = model.beam_search_position_selector(
                                start_cause_logits=start_cause_predictions[i],
                                end_cause_logits=end_cause_predictions[i],
                                start_effect_logits=start_effect_predictions[i],
                                end_effect_logits=end_effect_predictions[i],
                                attention_mask=batch["attention_mask"][i],
                                word_ids=word_ids,     
                                topk=args.topk,
                            )
                    else:
                        start_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        start_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    
                        start_cause_predictions[i][0] = -1e4
                        end_cause_predictions[i][0] = -1e4
                        start_effect_predictions[i][0] = -1e4
                        end_effect_predictions[i][0] = -1e4

                        start_cause_predictions[i][len(word_ids) - 1] = -1e4
                        end_cause_predictions[i][len(word_ids) - 1] = -1e4
                        start_effect_predictions[i][len(word_ids) - 1] = -1e4
                        end_effect_predictions[i][len(word_ids) - 1] = -1e4

                        start_cause = start_cause_predictions[i].argmax().item()
                        end_cause = end_cause_predictions[i].argmax().item()
                        start_effect = start_effect_predictions[i].argmax().item()
                        end_effect = end_effect_predictions[i].argmax().item()
                    
                    has_signal = 1
                    if args.signal_classification:
                        if not args.pretrained_signal_detector:
                            has_signal = outputs["signal_classification_logits"][i].argmax().item()
                        else:
                            has_signal = signal_detector.predict(text=batch["text"][i])

                    if has_signal:
                        start_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4

                        start_signal_predictions[i][0] = -1e4
                        end_signal_predictions[i][0] = -1e4

                        start_signal_predictions[i][len(word_ids) - 1] = -1e4
                        end_signal_predictions[i][len(word_ids) - 1] = -1e4

                        start_signal = start_signal_predictions[i].argmax().item()
                        end_signal_predictions[i][: start_signal] = -1e4
                        end_signal_predictions[i][start_signal + 5: ] = -1e4
                        end_signal = end_signal_predictions[i].argmax().item()

                    if not args.beam_search:
                        space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + space_splitted_tokens[word_ids[start_cause]]
                        space_splitted_tokens[word_ids[end_cause]] = space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + space_splitted_tokens[word_ids[start_effect]]
                        space_splitted_tokens[word_ids[end_effect]] = space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                        
                        if has_signal:
                            space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + space_splitted_tokens[word_ids[start_signal]]
                            space_splitted_tokens[word_ids[end_signal]] = space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        
                        predictions.append([' '.join(space_splitted_tokens)] * 2)
                    else:
                        start_cause, end_cause, start_effect, end_effect = indices1

                        this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                        this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                        this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                        this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'

                        if has_signal:
                            this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                            this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        generated_sentence_1 = ' '.join(this_space_splitted_tokens)

                        start_cause, end_cause, start_effect, end_effect = indices2

                        this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                        this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                        this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                        this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                        
                        if has_signal:
                            this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                            this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        generated_sentence_2 = ' '.join(this_space_splitted_tokens)

                        predictions.append([generated_sentence_1, generated_sentence_2])

            main_results = evaluate(truth, predictions)

            best_overall_f1 = max(best_overall_f1, main_results["Overall"]["f1"])
            best_cause_f1 = max(best_cause_f1, main_results["Cause"]["f1"])
            best_effect_f1 = max(best_effect_f1, main_results["Effect"]["f1"])
            best_signal_f1 = max(best_signal_f1, main_results["Signal"]["f1"])

            logger.info("Cause | P: {} | R: {} | F1: {}".format(main_results["Cause"]["precision"], main_results["Cause"]["recall"], main_results["Cause"]["f1"]))
            logger.info("Effect | P: {} | R: {} | F1: {}".format(main_results["Effect"]["precision"], main_results["Effect"]["recall"], main_results["Effect"]["f1"]))
            logger.info("Signal | P: {} | R: {} | F1: {}".format(main_results["Signal"]["precision"], main_results["Signal"]["recall"], main_results["Signal"]["f1"]))
            logger.info("Overall | P: {} | R: {} | F1: {}".format(main_results["Overall"]["precision"], main_results["Overall"]["recall"], main_results["Overall"]["f1"]))
            logger.info("best-overall-f1: {}".format(best_overall_f1))

            wandb.log(
                {
                    "Best-Overall-F1": best_overall_f1,
                    "Best-Cause-F1": best_cause_f1,
                    "Best-Effect-F1": best_effect_f1,
                    "Best-Signal-F1": best_signal_f1,
                    "Cause-Precision": main_results["Cause"]["precision"],
                    "Cause-Recall": main_results["Cause"]["recall"],
                    "Cause-F1": main_results["Cause"]["f1"],
                    "Effect-Precision": main_results["Effect"]["precision"],
                    "Effect-Recall": main_results["Effect"]["recall"],
                    "Effect-F1": main_results["Effect"]["f1"],
                    "Signal-Precision": main_results["Signal"]["precision"],
                    "Signal-Recall": main_results["Signal"]["recall"],
                    "Signal-F1": main_results["Signal"]["f1"],
                    "Overall-Precision": main_results["Overall"]["precision"],
                    "Overall-Recall": main_results["Overall"]["recall"],
                    "Overall-F1": main_results["Overall"]["f1"],
                }
            )

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

            with open(f"{output_dir}/predictions.txt", "w") as f:
                for row in truth.iterrows():
                    answers = eval(row[1]["causal_text_w_pairs"])
                    if len(answers) > 0:
                        f.write("Answers:\n")
                        for i, answer in enumerate(answers):
                            f.write(f"{i}. {answer}\n")
                        f.write("\n")
                        f.write("Prediction:\n")
                        f.write(f"{predictions[row[0]]}\n")
                        f.write("===============================\n")
            
            # export result of every epoch to the file
            with open(f"{output_dir}/submission.json", "w") as f:
                for i, prediction in enumerate(predictions):
                    f.write(json.dumps({"index": i, "prediction": prediction}) + "\n")

            # export best result to the file
            if best_overall_f1 == main_results["Overall"]["f1"]:
                with open(f"{args.output_dir}/best-submission.json", "w") as f:
                    for i, prediction in enumerate(predictions):
                        f.write(json.dumps({"index": i, "prediction": prediction}) + "\n")
    else:
        assert args.load_checkpoint_for_test is not None
        model.load_state_dict(torch.load(args.load_checkpoint_for_test))

        model.eval()
        predictions = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**{k: batch[k] if k in batch else None for k in ["input_ids", "attention_mask", "token_type_ids", "signal_bias_mask"]})

            start_cause_predictions = outputs["start_arg0_logits"]
            end_cause_predictions = outputs["end_arg0_logits"]

            start_effect_predictions = outputs["start_arg1_logits"]
            end_effect_predictions = outputs["end_arg1_logits"]

            start_signal_predictions = outputs["start_sig_logits"]
            end_signal_predictions = outputs["end_sig_logits"]

            for i in range(len(batch["input_ids"])):
                word_ids = batch["word_ids"][i]
                space_splitted_tokens = batch["text"][i].split(" ")

                if args.postprocessing_position_selector:
                    if not args.beam_search:
                        start_cause, end_cause, start_effect, end_effect = model.position_selector(
                            start_cause_logits=start_cause_predictions[i],
                            end_cause_logits=end_cause_predictions[i],
                            start_effect_logits=start_effect_predictions[i],
                            end_effect_logits=end_effect_predictions[i],
                            attention_mask=batch["attention_mask"][i],
                            word_ids=word_ids,
                        )
                    else:
                        indices1, indices2, score1, score2, topk_scores = model.beam_search_position_selector(
                            start_cause_logits=start_cause_predictions[i],
                            end_cause_logits=end_cause_predictions[i],
                            start_effect_logits=start_effect_predictions[i],
                            end_effect_logits=end_effect_predictions[i],
                            attention_mask=batch["attention_mask"][i],
                            word_ids=word_ids,     
                        )
                else:
                    start_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    end_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    start_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    end_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                
                    start_cause_predictions[i][0] = -1e4
                    end_cause_predictions[i][0] = -1e4
                    start_effect_predictions[i][0] = -1e4
                    end_effect_predictions[i][0] = -1e4

                    start_cause_predictions[i][len(word_ids) - 1] = -1e4
                    end_cause_predictions[i][len(word_ids) - 1] = -1e4
                    start_effect_predictions[i][len(word_ids) - 1] = -1e4
                    end_effect_predictions[i][len(word_ids) - 1] = -1e4

                    start_cause = start_cause_predictions[i].argmax().item()
                    end_cause = end_cause_predictions[i].argmax().item()
                    start_effect = start_effect_predictions[i].argmax().item()
                    end_effect = end_effect_predictions[i].argmax().item()
                
                has_signal = 1
                if args.signal_classification:
                    if not args.pretrained_signal_detector:
                        has_signal = outputs["signal_classification_logits"][i].argmax().item()
                    else:
                        has_signal = signal_detector.predict(text=batch["text"][i])

                if has_signal:
                    start_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    end_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4

                    start_signal_predictions[i][0] = -1e4
                    end_signal_predictions[i][0] = -1e4

                    start_signal_predictions[i][len(word_ids) - 1] = -1e4
                    end_signal_predictions[i][len(word_ids) - 1] = -1e4

                    start_signal = start_signal_predictions[i].argmax().item()
                    end_signal_predictions[i][: start_signal] = -1e4
                    end_signal_predictions[i][start_signal + 5: ] = -1e4
                    end_signal = end_signal_predictions[i].argmax().item()

                if not args.beam_search:
                    space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + space_splitted_tokens[word_ids[start_cause]]
                    space_splitted_tokens[word_ids[end_cause]] = space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                    space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + space_splitted_tokens[word_ids[start_effect]]
                    space_splitted_tokens[word_ids[end_effect]] = space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                    
                    if has_signal:
                        space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + space_splitted_tokens[word_ids[start_signal]]
                        space_splitted_tokens[word_ids[end_signal]] = space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                    
                    predictions.append([' '.join(space_splitted_tokens)] * 2)
                else:
                    start_cause, end_cause, start_effect, end_effect = indices1

                    this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                    this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                    this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                    this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                    this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'

                    if has_signal:
                        this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                        this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                    generated_sentence_1 = ' '.join(this_space_splitted_tokens)

                    start_cause, end_cause, start_effect, end_effect = indices2

                    this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                    this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                    this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                    this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                    this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                    
                    if has_signal:
                        this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                        this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                    generated_sentence_2 = ' '.join(this_space_splitted_tokens)

                    predictions.append([generated_sentence_1, generated_sentence_2])

        # export result of every epoch to the file
        with open(f"/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/testset-submission-{datetime.now()}.json", "w") as f:
            for i, prediction in enumerate(predictions):
                f.write(json.dumps({"index": i, "prediction": prediction}) + "\n")

    wandb.finish()


if __name__ == "__main__":
    main()