"""
data_helper.py
 - utility functions to process data (CoQA)
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
from qa_util import _improve_answer_span

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class CoQAExample(object):
    """
    a single training/test example for CoQA dataset.
    """
    def __init__(self,
                 paragraph_id,
                 turn_id,
                 question_texts,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 yes_no_flag=None,
                 yes_no_ans=None):
        self.paragraph_id = paragraph_id
        self.turn_id = turn_id
        self.question_texts = question_texts # list of question_text string, sorted in time order
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.yes_no_flag = yes_no_flag
        self.yes_no_ans = yes_no_ans



class ExampleFeature(object):
    """
    feature for one example - no chunking of documents
    """
    def __init__(self,
                 example_index,
                 query_tokens,
                 doc_tokens,
                 tok_to_orig_map,
                 start_position=None,
                 end_position=None,
                 yes_no_flag=None,
                 yes_no_ans=None):
        self.example_index = example_index
        self.query_tokens = query_tokens
        self.doc_tokens = doc_tokens
        # map position of tokenized doc_tokens to position in orignal doc_tokens
        self.tok_to_orig_map=tok_to_orig_map
        self.start_position = start_position
        self.end_position = end_position
        self.yes_no_flag = yes_no_flag
        self.yes_no_ans = yes_no_ans



def read_coqa_examples(input_file, is_training=True, use_history=False, n_history=-1):
    """
    read a CoQA json file into a list of QA examples
    """
    total_cnt = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
        
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        # process story text
        paragraph_text = entry["story"]
        paragraph_id = entry["id"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            # each char is mapped to word position
            char_to_word_offset.append(len(doc_tokens) - 1)

        # process questions
        question_history_texts = []
        for (question, ans) in zip(entry['questions'], entry['answers']):
            total_cnt += 1  
            cur_question_text = question["input_text"]
            question_history_texts.append(cur_question_text)
            question_id = question["turn_id"]
            ans_id = ans["turn_id"]
            start_position = None
            end_position =None
            yes_no_flag = None
            yes_no_ans = None
            orig_answer_text = None
            if (question_id != ans_id):
                print("question turns are not ordered!")
                print("mismatched question {}".format(cur_question_text))
            if is_training:
                orig_answer_text = ans["text"]
                answer_offset = ans["span_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                if (answer_offset+answer_length >= len(char_to_word_offset)):
                    end_position = char_to_word_offset[-1]
                else:
                    end_position = char_to_word_offset[answer_offset + answer_length]
                actual_text = " ".join(doc_tokens[start_position:(end_position+1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                yes_no_flag = int(ans["yes_no_flag"])
                yes_no_ans = int(ans["yes_no_ans"])
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                    continue

            if (use_history):
                if (n_history == -1 or n_history > len(question_history_texts)):
                    question_texts = question_history_texts[:]
                else:
                    question_texts = question_history_texts[-1*n_history:]
            else:
                question_texts = question_history_texts[-1]
            
            example = CoQAExample(
                paragraph_id=paragraph_id,
                turn_id=question_id,
                question_texts=question_texts,
                doc_tokens=doc_tokens,
                orig_answer_text = orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                yes_no_flag=yes_no_flag,
                yes_no_ans=yes_no_ans)
            examples.append(example)
    logger.info("Total raw examples: {}".format(total_cnt))
    return examples


def convert_examples_to_features(examples, tokenizer, max_query_length,
                                 is_training, append_history):
    """
    @input format:
    if append_history is True, query_tokens=[query, prev_queries,]
    if append_history is False, query_tokens= [prev_queries, query]
    """
    features = []
    for (example_index, example) in enumerate(examples):
        all_query_tokens = [tokenizer.tokenize(question_text) for question_text in example.question_texts]
        # same as basic Bert
        if append_history:
            all_query_tokens = all_query_tokens[::-1]
        flat_all_query_tokens = []
        for query_tokens in all_query_tokens:
            flat_all_query_tokens += query_tokens
        if append_history:
            query_tokens = flat_all_query_tokens[:max_query_length]
        else:
            query_tokens = flat_all_query_tokens[-1*max_query_length:]
            
        # doc_tokens
        tok_to_orig_index = []
        # tok_to_orig_map:
        # map the token position in tokenized all_doc_tokens to
        # the token position of original text by doc_tokens
        tok_to_orig_map = {}
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            # the orig word is mapped to its first sub token
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_map[len(all_doc_tokens)] = i
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # start/end position
        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                # tok_end_position is the last sub token of orig end_position
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        features.append(
            ExampleFeature(
                example_index=example_index,
                query_tokens=query_tokens,
                doc_tokens=all_doc_tokens,
                tok_to_orig_map=tok_to_orig_map,
                start_position=tok_start_position,
                end_position=tok_end_position,
                yes_no_flag=example.yes_no_flag,
                yes_no_ans=example.yes_no_ans))
    return features


RawResult = collections.namedtuple("RawResult", \
                                   ["example_index", "stop_logits", "start_logits", "end_logits", \
                                   "yes_no_flag_logits", "yes_no_ans_logits", "id_to_tok_map"])


def make_predictions(all_examples, all_features, all_results, n_best_size, \
                     max_answer_length, do_lower_case, \
                     verbose_logging, validate_flag=True):
    
    assert len(all_examples) == len(all_features)
    
    example_index_to_results = collections.defaultdict(list)
    for result in all_results:
        example_index_to_results[result.example_index].append(result)
    
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["result_index", "start_index", "end_index", "text", "logprob"])

    validate_predictions = dict()
    all_predictions = []
    all_nbest_json = []
    for (example_index, feature) in enumerate(all_features):
        example = all_examples[example_index]
        results = example_index_to_results[example_index]
        prelim_predictions = []
        for result_index, result in enumerate(results):
            #stop_logprob = np.log(result.stop_prob)
            #yes_no_flag_logprobs = np.log(_compute_softmax(result.yes_no_flag_logits)) # (2,)
            #yes_no_ans_logprobs = np.log(_compute_softmax(result.yes_no_ans_logits)) # (2,)
            
            # yes-no question
            if (np.argmax(result.yes_no_flag_logits) == 1):
                if (np.argmax(result.yes_no_ans_logits) == 1):
                    text = 'yes'
                    #logprob = stop_logprob + yes_no_flag_logprobs[1] + yes_no_ans_logprobs[1]
                    logprob = result.stop_logits[1] + result.yes_no_flag_logits[1] + result.yes_no_ans_logits[1]
                else:
                    text = 'no'
                    #logprob = stop_logprob + yes_no_flag_logprobs[1] + yes_no_ans_logprobs[0]
                    logprob = result.stop_logits[1] + result.yes_no_flag_logits[1] + result.yes_no_ans_logits[0]
                prelim_predictions.append(
                    _PrelimPrediction(
                        result_index=result_index,
                        start_index=-1,
                        end_index=-1,
                        text=text,
                        logprob=logprob))
                continue
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            #start_logprobs = np.log(_compute_softmax(result.start_logits))
            #end_logprobs = np.log(_compute_softmax(result.end_logits))

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index not in result.id_to_tok_map:
                        continue
                    if end_index not in result.id_to_tok_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    #logprob = stop_logprob + yes_no_flag_logprobs[0] + \
                    #          start_logprobs[start_index] + end_logprobs[end_index]
                    logprob = result.stop_logits[1] + result.yes_no_flag_logits[0] + \
                              result.start_logits[start_index] + result.end_logits[end_index]
                    prelim_predictions.append(
                        _PrelimPrediction(
                            result_index=result_index,
                            start_index=start_index,
                            end_index=end_index,
                            text=None,
                            logprob=logprob))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.logprob,
            reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "logprob"])
        
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            result = results[pred.result_index]
            if (pred.start_index == -1 or pred.end_index == -1):
                final_text = pred.text
            else:
                # answer_tokens: tokenized answers
                doc_start = result.id_to_tok_map[pred.start_index]
                doc_end = result.id_to_tok_map[pred.end_index]
                answer_tokens = feature.doc_tokens[doc_start:doc_end+1]
                answer_text = " ".join(answer_tokens)
                # De-tokenize WordPieces that have been split off.
                answer_text = answer_text.replace(" ##", "")
                answer_text = answer_text.replace("##", "")
                # Clean whitespace
                answer_text = answer_text.strip()
                answer_text = " ".join(answer_text.split())

                # orig_answer_tokens: original answers
                orig_doc_start = feature.tok_to_orig_map[doc_start]
                orig_doc_end = feature.tok_to_orig_map[doc_end]
                orig_answer_tokens = example.doc_tokens[orig_doc_start:orig_doc_end+1]
                orig_answer_text =  " ".join(orig_answer_tokens)
                
                # combine tokenized answer text and original text
                final_text = get_final_text(answer_text, orig_answer_text, do_lower_case, verbose_logging)
                
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    logprob=pred.logprob))
            if validate_flag:
                break

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", logprob=0.0))

        assert len(nbest) >= 1

        if validate_flag:
            validate_predictions[(example.paragraph_id, example.turn_id)] = nbest[0].text
        else:
            total_scores = []
            for entry in nbest:
                total_scores.append(entry.logprob)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["logprob"] = entry.logprob
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            cur_prediction = collections.OrderedDict()
            cur_prediction["id"] = example.paragraph_id
            cur_prediction["turn_id"] = example.turn_id
            cur_prediction["answer"] = nbest_json[0]["text"]
            all_predictions.append(cur_prediction)

            cur_nbest_json = collections.OrderedDict()
            cur_nbest_json["id"] = example.paragraph_id
            cur_nbest_json["turn_id"] = example.turn_id
            cur_nbest_json["answers"] = nbest_json
            all_nbest_json.append(cur_nbest_json)

    if validate_flag:
        return validate_predictions
    else:
        return all_predictions, all_nbest_json


def write_predictions(all_examples, all_features, all_results, n_best_size, \
                      max_answer_length, do_lower_case, \
                      output_prediction_file, output_nbest_file, verbose_logging):
    
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    all_predictions, all_nbest_json = make_predictions(all_examples, all_features, \
                                                       all_results, n_best_size, \
                                                       max_answer_length, do_lower_case, \
                                                       verbose_logging, validate_flag=False)

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")






