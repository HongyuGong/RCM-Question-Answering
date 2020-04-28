"""
Bert model with yes-no flag and answer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForQA(BertPreTrainedModel):
    def __init__(self, config, allow_yes_no=False):
        super(BertForQA, self).__init__(config)
        self.bert = BertModel(config)
        self.allow_yes_no = allow_yes_no
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if allow_yes_no:
            self.yes_no_flag_outputs = nn.Linear(config.hidden_size, 2)
            self.yes_no_ans_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None,
                yes_no_flags=None, yes_no_answers=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)       
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if self.allow_yes_no:
            yes_no_flag_logits = self.yes_no_flag_outputs(sequence_output)
            yes_no_ans_logits = self.yes_no_ans_outputs(sequence_output)
            yes_no_flag_logits = yes_no_flag_logits.squeeze(-1)
            yes_no_ans_logits = yes_no_ans_logits.squeeze(-1)
            
        
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
                
            if self.allow_yes_no and yes_no_flags is not None and \
               yes_no_answers is not None:
                if len(yes_no_flags.size()) > 1:
                    yes_no_flags = yes_no_flags.squeeze(-1)
                if len(yes_no_answers.size()) > 1:
                    yes_no_answers = yes_no_answers.squeeze(-1)
                # [all examples]: yes_no_flag_loss
                flag_loss_fct = CrossEntropyLoss(reduction='mean').cuda()
                yes_no_flag_loss = flag_loss_fct(yes_no_flag_logits, yes_no_flags)
                total_loss = 0.25 * yes_no_flag_loss

                # yes-no & wh- questions
                yes_no_indices = (yes_no_flags == 1).nonzero().view(-1)
                wh_indices = (yes_no_flags == 0).nonzero().view(-1)
                # [wh-questions] start_loss & end_loss
                selected_start_positions = start_positions.index_select(0, wh_indices)
                selected_end_positions = end_positions.index_select(0, wh_indices)
                selected_start_logits = start_logits.index_select(0, wh_indices)
                selected_end_logits = end_logits.index_select(0, wh_indices)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                # here index is word index instead of sample index
                ignored_index = selected_start_logits.size(1)
                selected_start_positions.clamp_(0, ignored_index)
                selected_end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index).cuda()
                if (selected_start_positions.size()[0] > 0):
                    start_loss = loss_fct(selected_start_logits, selected_start_positions)
                    end_loss = loss_fct(selected_end_logits, selected_end_positions)
                    total_loss += 0.25 * start_loss + 0.25 * end_loss

                # [yes-no questions] yes_no_answer_loss
                selected_yes_no_ans_logits = yes_no_ans_logits.index_select(0, yes_no_indices)
                selected_yes_no_answers = yes_no_answers.index_select(0, yes_no_indices)
                ans_loss_fct = CrossEntropyLoss(reduction='mean').cuda()
                if (selected_yes_no_ans_logits.size()[0] > 0):
                    yes_no_ans_loss = ans_loss_fct(selected_yes_no_ans_logits, \
                                                   selected_yes_no_answers)
                    total_loss += 0.25 * yes_no_ans_loss
                return total_loss
            else:
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index).cuda()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = 0.5 * start_loss + 0.5 * end_loss
                return total_loss
        
        else:
            if self.allow_yes_no:
                return start_logits, end_logits, yes_no_flag_logits, yes_no_ans_logits
            else:
                return start_logits, end_logits

