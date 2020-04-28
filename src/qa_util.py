"""
qa_util.py
 - utility functions for question answering
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_bert import whitespace_tokenize, BasicTokenizer, BertTokenizer
from optimization import BertAdam, warmup_linear
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
#from eval_triviaqa import evaluate_triviaqa


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def split_train_dev_data(examples, train_ratio=0.8):
    data_size = len(examples)
    train_size = int(data_size * train_ratio)
    train_inds = set(np.random.choice(range(data_size), train_size, replace=False))
    train_examples = [examples[idx] for idx in range(data_size) if idx in train_inds]
    dev_examples = [examples[idx] for idx in range(data_size) if idx not in train_inds]
    return train_examples, dev_examples
    
    
def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes



def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text



def gen_model_features(cur_global_pointers, batch_query_tokens, batch_doc_tokens, \
                       batch_start_positions, batch_end_positions, batch_max_doc_length, \
                       max_seq_length, tokenizer, is_train):
    # select next chunk doc_tokens
    chunk_doc_offsets = []
    chunk_doc_tokens = []
    chunk_start_positions = []
    chunk_end_positions = []
    chunk_stop_flags = []
    for index in range(len(cur_global_pointers)):
        # span: [doc_start, doc_span)
        doc_start = max(0, cur_global_pointers[index])
        doc_end = min(doc_start + batch_max_doc_length[index], len(batch_doc_tokens[index]))
        if (doc_start >= len(batch_doc_tokens[index])):
            doc_end = len(batch_doc_tokens[index])
            doc_start = max(0, doc_end - batch_max_doc_length[index])
        chunk_doc_tokens.append(batch_doc_tokens[index][doc_start:doc_end])
        chunk_doc_offsets.append(doc_start)
        if is_train:
            one_doc_len = doc_end - doc_start
            one_start_position = batch_start_positions[index] - doc_start
            one_end_position = batch_end_positions[index] - doc_start
            # for invalid span, set max_seq_length as index
            if (one_start_position < 0 or one_start_position >= one_doc_len or \
                one_end_position < 0 or one_end_position >= one_doc_len):
                chunk_stop_flags.append(0)
                chunk_start_positions.append(max_seq_length)
                chunk_end_positions.append(max_seq_length)
            else:
                chunk_stop_flags.append(1)
                chunk_start_positions.append(one_start_position)
                chunk_end_positions.append(one_end_position)

    # concat query and doc
    chunk_input_ids = []
    chunk_segment_ids = []
    chunk_input_mask = []
    # position in input_ids to position in batch_doc_tokens
    id_to_tok_maps = []
    for index in range(len(cur_global_pointers)):
        one_id_to_tok_map = {}
        one_query_tokens = batch_query_tokens[index]
        one_doc_tokens = chunk_doc_tokens[index]
        one_doc_offset = chunk_doc_offsets[index]
        one_tokens = []
        one_segment_ids = []
        one_tokens.append("[CLS]")
        one_segment_ids.append(0)
        # add query tokens
        for token in one_query_tokens:
            one_tokens.append(token)
            one_segment_ids.append(0)
        one_tokens.append("[SEP]")
        one_segment_ids.append(0)
        # add doc tokens
        for (i, token) in enumerate(one_doc_tokens):
            one_id_to_tok_map[len(one_tokens)] = one_doc_offset + i
            one_tokens.append(token)
            one_segment_ids.append(1)
        one_tokens.append("[SEP]")
        one_segment_ids.append(1)
        id_to_tok_maps.append(one_id_to_tok_map)

        # gen features
        one_input_ids = tokenizer.convert_tokens_to_ids(one_tokens)
        one_input_mask = [1] * len(one_input_ids)
        while len(one_input_ids) < max_seq_length:
            one_input_ids.append(0)
            one_input_mask.append(0)
            one_segment_ids.append(0)
        assert len(one_input_ids) == max_seq_length
        assert len(one_input_mask) == max_seq_length
        assert len(one_segment_ids) == max_seq_length
        chunk_input_ids.append(one_input_ids[:])
        chunk_input_mask.append(one_input_mask[:])
        chunk_segment_ids.append(one_segment_ids[:])
        if is_train:
            # adjust start_positions and end_positions due to doc offsets caused by query and CLS/SEP tokens in the input feature
            chunk_start_positions[index] += len(one_query_tokens) + 2
            chunk_end_positions[index] += len(one_query_tokens) + 2

    return chunk_input_ids, chunk_input_mask, chunk_segment_ids, id_to_tok_maps, \
           chunk_start_positions, chunk_end_positions, chunk_stop_flags
    


        





