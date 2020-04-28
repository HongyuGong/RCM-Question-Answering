"""
data_helper_trivia.py
 - helper functions to process trivia dataset
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


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class TriviaExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class ExampleFeature(object):
    def __init__(self,
                 example_index,
                 query_tokens,
                 doc_tokens,
                 tok_to_orig_map,
                 start_position=None,
                 end_position=None):
        self.example_index = example_index
        self.query_tokens = query_tokens
        self.doc_tokens = doc_tokens
        self.tok_to_orig_map = tok_to_orig_map
        self.start_position = start_position
        self.end_position = end_position
                 

def read_trivia_examples(input_file, is_training=True):
    total_cnt = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
        
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    no_answer_cnt = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
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
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if qa["answers"] == []:
                    no_answer_cnt += 1
                    continue
                if is_training:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    # word position
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    cleaned_start = actual_text.lower().find(cleaned_answer_text)
                    #if actual_text.find(cleaned_answer_text) == -1:
                    if cleaned_start == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                        continue
                    else:
                        # cleaned_answer_text might be lower cased, needs to be reconstructued from actual_text
                        orig_answer_text = actual_text[cleaned_start:cleaned_start+len(cleaned_answer_text)]
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""
                example = TriviaExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    print("# of questions without an answer".format(no_answer_cnt))
    return examples


def convert_examples_to_features(examples, tokenizer, max_query_length, is_training):
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        # mapping between orig and tok
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
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
                end_position=tok_end_position))
    return features


RawResult = collections.namedtuple("RawResult", \
                                   ["example_index", "stop_logits", \
                                   "start_logits", "end_logits", "id_to_tok_map"])

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
    all_predictions = collections.OrderedDict()
    all_nbest_json = []
    for (example_index, feature) in enumerate(all_features):
        example = all_examples[example_index]
        results = example_index_to_results[example_index]
        prelim_predictions = []
        for result_index, result in enumerate(results):
            #stop_logprob = np.log(result.stop_prob)
            #yes_no_flag_logprobs = np.log(_compute_softmax(result.yes_no_flag_logits)) # (2,)
            #yes_no_ans_logprobs = np.log(_compute_softmax(result.yes_no_ans_logits)) # (2,)
            
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
                    logprob = result.stop_logits[1] + result.start_logits[start_index] + \
                              result.end_logits[end_index]
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
            validate_predictions[example.qas_id] = nbest[0].text
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

            #cur_prediction = collections.OrderedDict()
            #cur_prediction["qid"] = example.qas_id
            #cur_prediction["answer"] = nbest_json[0]["text"]
            #all_predictions.append(cur_prediction)
            all_predictions[example.qas_id] = nbest_json[0]["text"]

            cur_nbest_json = collections.OrderedDict()
            cur_nbest_json["qid"] = example.qas_id
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

