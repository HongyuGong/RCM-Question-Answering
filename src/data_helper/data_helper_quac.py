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
from qa_util import _improve_answer_span, _get_best_indexes, get_final_text

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#UNK = "CANNOTANSWER"
followup_vocab = ['y', 'n', 'm']
#yesno_vocab = ['y', 'n', 'x']


class QuACExample(object):
    """
    a train/text example for QuAC dataset
    """
    def __init__(self,
               example_id,
               questions,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               yes_no_flag=None,
               yes_no_ans=None,
               followup=None):
        self.example_id = example_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.yes_no_flag = yes_no_flag
        self.yes_no_ans = yes_no_ans
        self.followup = followup

        
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
                 yes_no_ans=None,
                 followup=None):
        self.example_index = example_index
        self.query_tokens = query_tokens
        self.doc_tokens = doc_tokens
        # map position of tokenized doc_tokens to position in orignal doc_tokens
        self.tok_to_orig_map=tok_to_orig_map
        self.start_position = start_position
        self.end_position = end_position
        self.yes_no_flag = yes_no_flag
        self.yes_no_ans = yes_no_ans
        self.followup = followup


def read_quac_examples(input_file, is_training=True, use_history=False, n_history=-1):
    """
    read QuAC data into a list of QA examples
    """
    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)['data']

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    #yesno_symbols = set()
    #followup_symbols = set()
    for entry in input_data:
        para_obj = entry['paragraphs'][0]
        paragraph_id = para_obj['id']
        # process context paragraph
        paragraph_text = para_obj['context']
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
        for qa in para_obj['qas']:
            cur_question_text = qa['question']
            question_history_texts.append(cur_question_text)
            example_id = qa['id']
            # word position
            start_position = None
            end_position = None
            yes_no_flag = None
            followup = None
            orig_answer_text = None
            if is_training:
                answer = qa['answers'][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                if answer_offset + answer_length >= len(char_to_word_offset):
                    end_position = char_to_word_offset[-1]
                else:
                    end_position = char_to_word_offset[answer_offset + answer_length]
                actual_text = " ".join(doc_tokens[start_position:(end_position+1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                    continue
                #logger.info("yesno symbol: {}, followup symbol: {}".format(qa['yesno'], qa['followup']))
                yes_no_flag = int(qa['yesno'] in ['y','n'])
                yes_no_ans = int(qa['yesno'] == 'y')
                #yes_no_flag = yesno_vocab.index(qa['yesno'])
                #yesno_symbols.add(qa['yesno'])
                followup = followup_vocab.index(qa['followup'])
                #followup_symbols.add(qa['followup'])
            questions =  []
            if use_history:
                # !!! CONTINUE
                if n_history == -1 or len(question_history_texts) <= n_history:
                    questions = question_history_texts[:]
                else:
                    questions = question_history_texts[-1*n_history:]
            else:
                questions = [question_history_texts[-1]]
            example = QuACExample(
                example_id=example_id,
                questions=questions,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                yes_no_flag=yes_no_flag,
                yes_no_ans=yes_no_ans,
                followup=followup)
            examples.append(example)
        
    #logger.info("yesno symbols: {}, followup symbols: {}".format(yesno_symbols, followup_symbols))
    return examples


def convert_examples_to_features(examples, tokenizer, max_query_length, \
                                 is_training, append_history):
    features = []
    for (example_index, example) in enumerate(examples):
        all_query_tokens = [tokenizer.tokenize(question_text) for question_text in example.questions]
        cur_query_tokens = all_query_tokens[-1]
        prev_query_tokens = all_query_tokens[:-1]
        if append_history:
            prev_query_tokens = prev_query_tokens[::-1]
        flat_prev_query_tokens = []
        for query_tokens in prev_query_tokens:
            flat_prev_query_tokens += query_tokens

        if len(cur_query_tokens) + len(flat_prev_query_tokens) + 1 <= max_query_length:
            if append_history:
                query_tokens = cur_query_tokens + ['[SEP]'] + flat_prev_query_tokens
            else:
                query_tokens = flat_prev_query_tokens + ['[SEP]'] + cur_query_tokens
        else:
            prev_query_len = max_query_length  - 1 - len(cur_query_tokens)
            if append_history:
                query_tokens = cur_query_tokens + ['[SEP]'] + flat_prev_query_tokens[:prev_query_len]
            else:
                query_tokens = flat_prev_query_tokens[-1*prev_query_len:] + ['[SEP]'] + cur_query_tokens

        tok_to_orig_index = []
        tok_to_orig_map = {}
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
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
                yes_no_ans=example.yes_no_ans,
                followup=example.followup))
    return features


RawResult = collections.namedtuple("RawResult", \
                                   ["example_index", "stop_logits", \
                                   "start_logits", "end_logits", \
                                   "id_to_tok_map"])


def make_predictions(all_examples, all_features, all_results, n_best_size, \
                     max_answer_length, do_lower_case, verbose_logging, \
                     validate_flag=True):
    assert len(all_examples) == len(all_features)
    example_index_to_results = collections.defaultdict(list)
    for result in all_results:
        example_index_to_results[result.example_index].append(result)

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["result_index", "start_index", "end_index", "text", "logit"])

    validate_predictions = defaultdict(dict)
    all_predictions = []
    all_nbest_json = []
    for (example_index, feature) in enumerate(all_features):
        example = all_examples[example_index]
        results = example_index_to_results[example_index]
        prelim_predictions = []
        for result_index, result in enumerate(results):
            # yesno
            #yes_no_flag_logits = result.yes_no_flag_logits
            #yes_no_pred_flag = np.argmax(yes_no_flag_logits)

            # followup
            #followup_logits = result.followup_logits
            #followup = np.argmax(followup_logits)

            # answer span
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
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
                    """
                    logit = result.stop_logits[1] + yes_no_flag_logits[yes_no_pred_flag] + \
                            followup_logits[followup] + result.start_logits[start_index] + \
                            result.end_logits[end_index]
                    logit = result.stop_logits[1] + result.start_logits[start_index] + result.end_logits[end_index]
                    """
                    logit = result.stop_logits[1] + result.start_logits[start_index] + \
                        result.end_logits[end_index]
                    prelim_predictions.append(
                        _PrelimPrediction(
                            result_index=result_index,
                            start_index=start_index,
                            end_index=end_index,
                            text=None,
                            logit=logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.logit,
            reverse=True)

        _NbestPrediction = collections.namedtuple(
                    "NbestPrediction", ["text", "logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            result = results[pred.result_index]
            if pred.start_index < 0 or pred.end_index < 0:
                final_text = UNK
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
                    logit=pred.logit))
            if validate_flag:
                break

        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text=UNK,
                    logit=0.0))

        assert len(nbest) >= 1
        if validate_flag:
            qid = example.example_id
            dia_id = qid.split("_q#")[0]
            validate_predictions[dia_id][qid] = nbest[0].text

            total_scores = []
            for entry in nbest:
                total_scores.append(entry.logit)
            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["logit"] = entry.logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            
            cur_prediction = collections.OrderedDict()
            cur_prediction['example_id'] = example.example_id
            cur_prediction['answer'] = nbest_json[0]["text"]
            all_predictions.append(cur_prediction)

            cur_nbest_json = collections.OrderedDict()
            cur_nbest_json["example_id"] = example.example_id
            cur_nbest_json["answers"] = nbest_json
            all_nbest_json.append(cur_nbest_json)
            
    if validate_flag:
        return validate_predictions
    else:
        return all_predictions, all_nbest_json


def format_predictions(all_predictions, output_prediction_file):
    # format prediction outputs: https://s3.amazonaws.com/my89public/quac/example.json
    prediction_dict = defaultdict(list) # paragraph_id: (turn_id, example_id, yesno, answer, followup)
    for prediction in all_predictions:
        example_id = prediction['example_id']
        #yesno = prediction['yesno']
        answer = prediction['answer']
        #followup = prediction['followup']
        ids = example_id.split("_q#")
        paragraph_id = "".join(ids[:-1])
        turn_id = int(ids[-1])
        prediction_dict[paragraph_id].append((turn_id, example_id, answer))

    with open(output_prediction_file, "w") as writer:
        for paragraph_id in prediction_dict:
            predictions = prediction_dict[pragraph_id]
            sorted_predictions = sorted(predictions, key=lambda item:item[0], reverse=True)
            output_dict = OrderedDict()
            output_dict["best_span_str"] = [item[3] for item in sorted_predictions]
            output_dict["qid"] = [item[1] for item in sorted_predictions]
            #output_dict["yesno"] = [yesno_vocab[item[2]] for item in sorted_predictions]
            #output_dict["followup"] = [followup_vocab[item[4]] for item in sorted_predictions]
            writer.write(json.dumps(output_dict) + "\n")
    print("saving predictions to {}".format(output_prediction_file))



def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    all_predictions, all_nbest_json = make_predictions(all_examples, all_features, all_results, \
                                                       n_best_size, max_answer_length, do_lower_case, \
                                                       verbose_logging, validate_flag=False)
    
    format_predictions(all_predictions, output_prediction_file)

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

