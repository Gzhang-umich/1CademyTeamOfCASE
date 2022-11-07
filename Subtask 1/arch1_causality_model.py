import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
import json
from utils import *

NUM_LABELS = 2

class CausalityModel(transformers.PreTrainedModel):
    def __init__(self, entailment_model, event_detection_model):
        super().__init__(transformers.PretrainedConfig())
        self.entailment_model = entailment_model
        self.event_detection_model = event_detection_model
        self.fc_out = nn.Linear(NUM_LABELS, NUM_LABELS)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    @classmethod
    def create(cls, path_to_entailment, path_to_event_detection, path_to_tokenizer):

        entailment_model = AutoModelForSequenceClassification.from_pretrained(path_to_entailment)
        event_detection_model = AutoModelForSequenceClassification.from_pretrained(path_to_event_detection)
        tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
        attr_name = get_encoder_attr_name(entailment_model)
        shared_encoder = getattr(entailment_model, attr_name)
        setattr(event_detection_model, attr_name, shared_encoder)
        return cls(entailment_model=entailment_model, event_detection_model=event_detection_model, tokenizer=tokenizer)

    def forward(self, **kwargs):
        entailment_logits = self.entailment_model(**kwargs).logits
        event_detection_logits = self.event_detection_model(**kwargs).logits
        out = self.fc_out(entailment_logits + event_detection_logits)
        return out

    def training_step(self, **kwargs):
        logits = self.forward(**kwargs)
        loss = self.loss(logits, kwargs['labels']).mean()
        return loss
