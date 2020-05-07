"""
Recurrent Chunking Mechanism on BERT model
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
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from torch.distributions.categorical import Categorical


class stopNetwork(nn.Module):
    """
    input: chunk_states (bsz, hidden_size)
    output: stop logits (bsz, 2) -- 0: move, 1: stop
    """
    def __init__(self, input_size):
        super(stopNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, chunk_states):
        stop_logits = self.fc(chunk_states)
        return stop_logits


class moveStrideNetwork(nn.Module):
    """
    input: hidden_states (bsz, hidden_size)
    output: action probability (bsz, num_action_class)
    Function: sample an action and also give the action probability
    """
    def __init__(self, input_size, num_action_classes):
        super(moveStrideNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_action_classes)

    def forward(self, chunk_states, scheme="sample"):
        # stride_probs: (bsz, num_stride_choices)
        outputs = self.fc(chunk_states)
        stride_probs = F.softmax(outputs, dim=1)
        stride_log_probs = F.log_softmax(outputs, dim=1)
        if scheme == "sample":
            policy = Categorical(stride_probs.detach())
            # sampled_stride_inds: (bsz,)
            sampled_stride_inds = policy.sample()
        elif scheme == "greedy":
            # sampled_stride_inds: (bsz,)
            sampled_stride_inds = torch.argmax(stride_probs.detach(), dim=1)
        # sampled_stride_log_probs: (bsz, )
        sampled_stride_log_probs = stride_log_probs.gather(1, sampled_stride_inds.unsqueeze(1)).squeeze(1)
        return sampled_stride_inds, sampled_stride_log_probs


class recurLSTMNetwork(nn.Module):
    """
    lstm for recurrence
    """
    def __init__(self, input_size, hidden_size):
        super(recurLSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        print("LSTM recurrence...")

    def forward(self, x_t, lstm_prev_states):
        """
        Input
        x_t: (bsz, input_size)
        lstm_prev_states: h (bsz, hidden_size), c (bsz, hidden_size)
        Output
        h, c
        """
        bsz = x_t.size(0)
        if lstm_prev_states is None:
            lstm_prev_states = (torch.zeros([bsz, self.hidden_size], device=x_t.device), \
                                torch.zeros([bsz, self.hidden_size], device=x_t.device))
        
        hidden_states, cell_states = self.lstm(x_t, lstm_prev_states)
        return (hidden_states, cell_states)


class recurGatedNetwork(nn.Module):
    """
    Simplified version:
    hidden_states = x_t + prev_hidden_states
    """
    def __init__(self, input_size, hidden_size):
        super(recurGatedNetwork, self).__init__()
        self.attn = nn.Linear(input_size + hidden_size, 2)
        print("Gated recurrence...")

    def forward(self, x_t, prev_hidden_states):
        # weights: (bsz, 2)
        weights = F.softmax(self.attn(torch.cat([x_t, prev_hidden_states], dim=1)), dim=1)
        # weights: (bsz, 1, 2)
        weights = weights.unsqueeze(1)
        # cat_hidden_states: (bsz, 2, hidden_size)
        cat_hidden_states = torch.cat([x_t.unsqueeze(1), prev_hidden_states.unsqueeze(1)], dim=1)
        # hidden_states: (bsz, hidden_size)
        hidden_states = torch.matmul(weights, cat_hidden_states).squeeze(1)
        return hidden_states


class RCMBert(BertPreTrainedModel):
    def __init__(self, config, action_num, recur_type="gated", allow_yes_no=False):
        super(RCMBert, self).__init__(config)
        self.bert = BertModel(config)
        self.recur_type = recur_type
        self.allow_yes_no = allow_yes_no
        if recur_type == "gated":
            self.recur_network = recurGatedNetwork(config.hidden_size, config.hidden_size)
        elif recur_type == "lstm":
            self.recur_network = recurLSTMNetwork(config.hidden_size, config.hidden_size)
        else:
            print("Invalid recur_type: {}".format(recur_type))
            sys.exit(0)
        self.action_num = action_num
        self.stop_network = stopNetwork(config.hidden_size)
        self.move_stride_network = moveStrideNetwork(config.hidden_size, self.action_num)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.allow_yes_no:
            self.yes_no_flag_outputs = nn.Linear(config.hidden_size, 2)
            self.yes_no_ans_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.init_weights()


    def forward(self, input_ids, token_type_ids, attention_mask,
                prev_hidden_states, stop_flags=None,
                start_positions=None, end_positions=None,
                yes_no_flags=None, yes_no_answers=None):
        """
        Input:
        new chunk: input_ids, token_type_ids, attention_mask (bsz, chunk_len)
        stop_flags: whether the current chunk contains the answer_span
        prev_hidden_states: (bsz, hidden_size)
        Output:
        stop_probs, move_probs, start_logits, end_logits, yes_no_flag_logits, yes_no_ans_logits,
        prev_hidden_states
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # add dropout
        sequence_output = self.dropout(sequence_output)
        # sent_output: (batch_size, hidden_size)
        sent_output = sequence_output.narrow(1, 0, 1)
        sent_output = sent_output.squeeze(1)
        
        # combine hidden_states for stop and moving prediction
        if self.recur_type == "gated":
            cur_hidden_states = sent_output if prev_hidden_states is None \
                                else self.recur_network(sent_output, prev_hidden_states)
            recur_sent_output = cur_hidden_states
        elif self.recur_type == "lstm":
            cur_hidden_states = self.recur_network(sent_output, prev_hidden_states)
            recur_sent_output = cur_hidden_states[0]

        # stop logits: (bsz, 2)
        stop_logits = self.stop_network(recur_sent_output)
        
        # answer prediction in the current span
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # yes-no question
        if self.allow_yes_no:
            yes_no_flag_logits = self.yes_no_flag_outputs(sent_output)
            yes_no_ans_logits = self.yes_no_ans_outputs(sent_output)
            yes_no_flag_logits = yes_no_flag_logits.squeeze(-1)
            yes_no_ans_logits = yes_no_ans_logits.squeeze(-1)
        
        # get loss for stop & answer prediction in the chunk level
        if start_positions is not None and end_positions is not None and \
           stop_flags is not None:
            # stride 
            sampled_stride_inds, sampled_stride_log_probs = self.move_stride_network(recur_sent_output, scheme="sample")
            
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(stop_flags.size()) > 1:
                stop_flags = stop_flags.squeeze(-1)

            # stop loss
            stop_loss_fct = CrossEntropyLoss(reduction='mean')
            stop_loss = stop_loss_fct(stop_logits, stop_flags)

            # answer loss
            if self.allow_yes_no and yes_no_flags is not None and \
               yes_no_answers is not None:
                # ground truth
                if len(yes_no_flags.size()) > 1:
                    yes_no_flags = yes_no_flags.squeeze(-1)
                if len(yes_no_answers.size()) > 1:
                    yes_no_answers = yes_no_answers.squeeze(-1)
                    
                # for all samples: classify yes-no / wh- question
                # this is purely query-dependent, and not influenced by stop_flags
                flag_loss_fct = CrossEntropyLoss(reduction='mean')
                yes_no_flag_loss = flag_loss_fct(yes_no_flag_logits, yes_no_flags)
                answer_loss = 0.25 * yes_no_flag_loss
                
                # estimate loss only when the current chunk contains the answer
                yes_no_indices = (stop_flags + yes_no_flags == 2).nonzero().view(-1)
                wh_indices = (stop_flags - yes_no_flags == 1).nonzero().view(-1)
                # for samples with wh- questions
                selected_start_positions = start_positions.index_select(0, wh_indices)
                selected_end_positions = end_positions.index_select(0, wh_indices)
                selected_start_logits = start_logits.index_select(0, wh_indices)
                selected_end_logits = end_logits.index_select(0, wh_indices)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                # here index is word index instead of sample index
                ignored_index = selected_start_logits.size(1)
                selected_start_positions.clamp_(0, ignored_index)
                selected_end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                if (selected_start_positions.size(0) > 0):
                    start_loss = loss_fct(selected_start_logits, selected_start_positions)
                    end_loss = loss_fct(selected_end_logits, selected_end_positions)
                    answer_loss += 0.25 * start_loss + 0.25 * end_loss

                # for samples with yes-no questions
                # CrossEntropyLoss: input: (seq_len, C), target: (seq_len, )
                selected_yes_no_ans_logits = yes_no_ans_logits.index_select(0, yes_no_indices)
                selected_yes_no_answers = yes_no_answers.index_select(0, yes_no_indices)
                ans_loss_fct = CrossEntropyLoss(reduction='mean')
                if (selected_yes_no_ans_logits.size(0) > 0):
                    yes_no_ans_loss = ans_loss_fct(selected_yes_no_ans_logits, \
                                                   selected_yes_no_answers)
                    answer_loss += 0.25 * yes_no_ans_loss
                return stop_logits, sampled_stride_inds, sampled_stride_log_probs, \
                       start_logits, end_logits, yes_no_flag_logits, yes_no_ans_logits, \
                       cur_hidden_states, stop_loss, answer_loss
                    
            else:
                # only answer span selection
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                answer_loss = 0.5 * start_loss + 0.5 * end_loss
                
                return stop_logits, sampled_stride_inds, sampled_stride_log_probs, \
                       start_logits, end_logits, cur_hidden_states, stop_loss, answer_loss

        else:
            # stride 
            sampled_stride_inds, sampled_stride_log_probs = self.move_stride_network(recur_sent_output, scheme="greedy")
            if self.allow_yes_no:
                return stop_logits, sampled_stride_inds, sampled_stride_log_probs, \
                       start_logits, end_logits, yes_no_flag_logits, yes_no_ans_logits, \
                       cur_hidden_states
            else:
                return stop_logits, sampled_stride_inds, sampled_stride_log_probs, \
                       start_logits, end_logits, cur_hidden_states
        

