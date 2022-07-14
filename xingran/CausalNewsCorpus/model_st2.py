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


class ST2ModelV2(nn.Module):
    def __init__(self, args):
        super(ST2ModelV2, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path)

        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 6)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,     # [batch_size, 3]
        end_positions: Optional[torch.Tensor] = None,       # [batch_size, 3]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
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

        logits = self.classifier(sequence_output)
        start_arg0_logits, end_arg0_logits, start_arg1_logits, end_arg1_logits, start_sig_logits, end_sig_logits = logits.split(1, dim=-1)
        start_arg0_logits = start_arg0_logits.squeeze(-1).contiguous()
        end_arg0_logits = end_arg0_logits.squeeze(-1).contiguous()
        start_arg1_logits = start_arg1_logits.squeeze(-1).contiguous()
        end_arg1_logits = end_arg1_logits.squeeze(-1).contiguous()
        start_sig_logits = start_sig_logits.squeeze(-1).contiguous()
        end_sig_logits = end_sig_logits.squeeze(-1).contiguous()

        # start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits = end_logits.squeeze(-1).contiguous()

        arg0_loss = None
        arg1_loss = None
        sig_loss = None
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()

            start_arg0_loss = loss_fct(start_arg0_logits, start_positions[:, 0])
            end_arg0_loss = loss_fct(end_arg0_logits, end_positions[:, 0])
            arg0_loss = (start_arg0_loss + end_arg0_loss) / 2

            start_arg1_loss = loss_fct(start_arg1_logits, start_positions[:, 1])
            end_arg1_loss = loss_fct(end_arg1_logits, end_positions[:, 1])
            arg1_loss = (start_arg1_loss + end_arg1_loss) / 2

            start_sig_loss = loss_fct(start_sig_logits, start_positions[:, 2])
            end_sig_loss = loss_fct(end_sig_logits, end_positions[:, 2])
            sig_loss = (start_sig_loss + end_sig_loss) / 2

            total_loss = (arg0_loss + arg1_loss + sig_loss) / 3
        
        return {
            'start_arg0_logits': start_arg0_logits,
            'end_arg0_logits': end_arg0_logits,
            'start_arg1_logits': start_arg1_logits,
            'end_arg1_logits': end_arg1_logits,
            'start_sig_logits': start_sig_logits,
            'end_sig_logits': end_sig_logits,
            'arg0_loss': arg0_loss,
            'arg1_loss': arg1_loss,
            'sig_loss': sig_loss,
            'loss': total_loss,
        }
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions = start_positions.clamp(0, ignored_index)
        #     end_positions = end_positions.clamp(0, ignored_index)

            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # start_loss = loss_fct(start_logits, start_positions)
            # end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )