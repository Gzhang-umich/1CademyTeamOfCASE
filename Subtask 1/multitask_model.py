import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
import logging
logging.basicConfig(level=logging.INFO)
from multitask_trainer import *
from utils import *

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, get_encoder_attr_name(model))
            else:
                setattr(model, get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

def get_multitask_models(model_name,
                        per_device_train_batch_size=8):

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def combine_sentences(example):
        example['sentence'] = example['sentence1'] + ' ' + example['sentence2']
        if example['label'] == 1:
            example['label'] = 0
        elif example['label'] == 0:
            example['label'] = 1
        return example

    def convert_to_features(example_batch):
        inputs = list(example_batch['sentence'])
        features = tokenizer.batch_encode_plus(
            inputs, max_length=128, pad_to_max_length=True
        )
        features["labels"] = example_batch["label"]
        return features

    dataset_dict = {"entailment": load_dataset('glue', "rte"),
                            "event_detection": load_dataset('adamnik/event_detection_dataset')}
    dataset_dict['entailment']['train'] = dataset_dict['entailment']['train'].map(lambda examples: combine_sentences(examples), remove_columns=['sentence1', 'sentence2', 'idx'])
    dataset_dict['entailment']['validation'] = dataset_dict['entailment']['validation'].map(lambda examples: combine_sentences(examples), remove_columns=['sentence1', 'sentence2', 'idx'])
    dataset_dict['entailment']['test'] = dataset_dict['entailment']['test'].map(lambda examples: combine_sentences(examples), remove_columns=['sentence1', 'sentence2', 'idx'])

    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "entailment": transformers.AutoModelForSequenceClassification,
            "event_detection": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "entailment": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
            "event_detection": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
        },
    )

    convert_func_dict = {
        "entailment": convert_to_features,
        "event_detection": convert_to_features,
    }

    columns_dict = {
        "entailment": ['input_ids', 'attention_mask', 'labels'],
        "event_detection": ['input_ids', 'attention_mask', 'labels'],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )

    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir="./models/multitask_model",
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=3,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )

    trainer.train()
    return multitask_model.taskmodels_dict['entailment'], multitask_model.taskmodels_dict['event_detection'], tokenizer
