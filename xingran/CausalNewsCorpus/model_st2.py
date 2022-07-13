import torch
from torch import nn

from typing import List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer, AutoConfig


class ST2Model(nn.Module):
    def __init__(self, args):
        super(ST2Model, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path)

        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.ce_classifier = nn.Linear(self.config.hidden_size, 5)
        self.sig_classifier = nn.Linear(self.config.hidden_size, 3)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        ce_labels: Optional[torch.LongTensor] = None,
        sig_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        ce_logits = self.ce_classifier(sequence_output)
        sig_logits = self.sig_classifier(sequence_output)

        ce_loss = None
        if ce_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(ce_logits.view(-1, 5), ce_labels.view(-1))
        
        sig_loss = None
        if sig_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sig_loss = loss_fct(sig_logits.view(-1, 3), sig_labels.view(-1))

        loss = None
        if ce_loss is not None and sig_loss is not None:
            loss = ce_loss + sig_loss
        return {
            'ce_logits': ce_logits,
            'sig_logits': sig_logits,
            'ce_loss': ce_loss,
            'sig_loss': sig_loss,
            'loss': loss,
        }