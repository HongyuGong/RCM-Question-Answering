"""
RCM-BERT for question answering on Trivia dataset
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
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from transformers.tokenization_bert import whitespace_tokenize, BasicTokenizer, BertTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from optimization import BertAdam, warmup_linear
from model.modeling_RCM import RCMBert
from model.rl_reward import reward_estimation, reward_estimation_for_stop
#from model.modeling_BERT import BertForQA
from data_helper.qa_util import split_train_dev_data, gen_model_features, _improve_answer_span, \
     get_final_text, _compute_softmax, _get_best_indexes
from data_helper.data_helper_trivia import read_trivia_examples, convert_examples_to_features, \
     RawResult, make_predictions, write_predictions
import data_helper.json_utils
import data_helper.trivia_dataset_utils
from eval_helper.eval_triviaqa import evaluate_triviaqa



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


stide_action_space = [-32, 64, 128, 256, 512]
# for sanity check
#stride_action_space = [128]

def validate_model(args, model, tokenizer, dev_examples, dev_features,
                   dev_dataloader, dev_ground_truth, device):
    all_results = []
    for dev_step, batch_dev_indices in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
        batch_dev_features = [dev_features[ind] for ind in batch_dev_indices]
        batch_query_tokens = [f.query_tokens for f in batch_dev_features]
        batch_doc_tokens = [f.doc_tokens for f in batch_dev_features]
        batch_size = len(batch_dev_features)
        cur_global_pointers = [0] * batch_size
        batch_max_doc_length = [args.max_seq_length-3-len(query_tokens) for query_tokens in batch_query_tokens]
        stop_probs = []
        prev_hidden_states = None
        for t in range(args.max_read_times):
            chunk_input_ids, chunk_input_mask, chunk_segment_ids, id_to_tok_maps, _, _, _ = \
                             gen_model_features(cur_global_pointers, batch_query_tokens, batch_doc_tokens, \
                                                None, None, batch_max_doc_length, args.max_seq_length, \
                                                tokenizer, is_train=False)
            chunk_input_ids = torch.tensor(chunk_input_ids, dtype=torch.long, device=device)
            chunk_input_mask = torch.tensor(chunk_input_mask, dtype=torch.long, device=device)
            chunk_segment_ids = torch.tensor(chunk_segment_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                chunk_stop_logits, chunk_stride_inds, chunk_stride_log_probs, \
                                   chunk_start_logits, chunk_end_logits, prev_hidden_states \
                                   = model(chunk_input_ids, chunk_segment_ids, chunk_input_mask, prev_hidden_states)
            chunk_stop_logits = chunk_stop_logits.detach().cpu().tolist()
            # stop_probs: current chunk contains answer
            #chunk_stop_probs = chunk_stop_probs.detach().cpu().tolist()

            # find top answer texts for the current chunk
            for i, example_index in enumerate(batch_eval_indices):
                stop_logits = chunk_stop_logits[i]
                start_logits = chunk_start_logits[i].detach().cpu().tolist()
                end_logits = chunk_end_logits[i].detach().cpu().tolist()
                #yes_no_flag_logits = chunk_yes_no_flag_logits[i].detach().cpu().tolist()
                #yes_no_ans_logits = chunk_yes_no_ans_logits[i].detach().cpu().tolist()
                id_to_tok_map = id_to_tok_maps[i]
                example_index = example_index.item()
                all_results.append(RawResult(example_index=example_index,
                                             stop_logits=stop_logits,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             id_to_tok_map=id_to_tok_map))

            # take movement action
            if args.supervised_pretraining:
                chunk_strides = [args.doc_stride] * batch_size
            else:
                chunk_strides = [stride_action_space[stride_ind] for stride_ind in chunk_stride_inds.tolist()]
            cur_global_pointers = [cur_global_pointers[ind] + chunk_strides[ind] for ind in range(len(cur_global_pointers))]
            # put pointer be put to 0 or the last doc token is it
            cur_global_pointers = [min(max(0, cur_global_pointers[ind]), len(batch_doc_tokens[ind])-1) \
                                   for ind in range(len(cur_global_pointers))]

    dev_predictions = make_predictions(dev_examples, dev_features, all_results, args.n_best_size, \
                                        args.max_answer_length, args.do_lower_case, \
                                        args.verbose_logging, validate_flag=True)
    dev_scores = evaluate_triviaqa(dev_ground_truth, dev_predictions)
    dev_score = dev_scores['f1']
    logger.info('step: {}, dev score: {}'.format(step, dev_score))
    if (dev_score > best_dev_score):
        best_model_to_save = model.module if hasattr(model, 'module') else model
        best_output_model_file = os.path.join(args.output_dir, "best_pretrained_model.bin")
        torch.save(best_model_to_save.state_dict(), best_output_model_file)
        best_dev_score = max(best_dev_score, dev_score)
        logger.info("Best dev score: {}, saved to best_pretrained_model.bin".format(dev_score))
        #log.write('Best eval score: '+str(best_eval_score)+'\n')


def train_model(args, model, tokenizer, optimizer, train_examples, train_features,
                dev_examples, dev_features, dev_ground_truth, device, n_gpu, t_total):
    train_indices = torch.arange(len(train_features), dtype=torch.long)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_indices)
    else:
        train_sampler = DistributedSampler(train_indices)
    train_dataloader = DataLoader(train_indices, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.do_validate:
        dev_indices = torch.arange(len(dev_features), dtype=torch.long)
        dev_sampler = SequentialSampler(dev_indices)
        dev_dataloader = DataLoader(dev_indices, sampler=dev_sampler, batch_size=args.predict_batch_size)
        
    best_dev_score = 0.0
    epoch = 0
    global_step = 0
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        training_loss = 0.0
        for step, batch_indices in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_features = [train_features[ind] for ind in batch_indices]
            batch_query_tokens = [f.query_tokens for f in batch_features]
            batch_doc_tokens = [f.doc_tokens for f in batch_features]
            batch_start_positions = [f.start_position for f in batch_features]
            batch_end_positions = [f.end_position for f in batch_features]

            batch_size = len(batch_features)
            cur_global_pointers = [0] * batch_size # global position of current pointer at the document
            batch_max_doc_length = [args.max_seq_length-3-len(query_tokens) for query_tokens in batch_query_tokens]

            stride_log_probs = []
            stop_rewards = []
            stop_probs = []
            stop_loss = None
            answer_loss = None
            prev_hidden_states = None
            for t in range(args.max_read_times):
                # features at the current chunk
                chunk_input_ids, chunk_input_mask, chunk_segment_ids, id_to_tok_maps, \
                                 chunk_start_positions, chunk_end_positions, chunk_stop_flags = \
                                 gen_model_features(cur_global_pointers, batch_query_tokens, batch_doc_tokens, \
                                                    batch_start_positions, batch_end_positions, batch_max_doc_length, \
                                                    args.max_seq_length, tokenizer, is_train=True)
                chunk_input_ids = torch.tensor(chunk_input_ids, dtype=torch.long, device=device)
                chunk_input_mask = torch.tensor(chunk_input_mask, dtype=torch.long, device=device)
                chunk_segment_ids = torch.tensor(chunk_segment_ids, dtype=torch.long, device=device)
                chunk_start_positions = torch.tensor(chunk_start_positions, dtype=torch.long, device=device)
                chunk_end_positions = torch.tensor(chunk_end_positions, dtype=torch.long, device=device)
                #chunk_yes_no_flags = torch.tensor(batch_yes_no_flags, dtype=torch.long, device=device)
                #chunk_yes_no_answers = torch.tensor(batch_yes_no_answers, dtype=torch.long, device=device)
                chunk_stop_flags = torch.tensor(chunk_stop_flags, dtype=torch.long, device=device)
                
                # model to find span
                chunk_stop_logits, chunk_stride_inds, chunk_stride_log_probs, \
                                   chunk_start_logits, chunk_end_logits, prev_hidden_states, \
                                   chunk_stop_loss, chunk_answer_loss = \
                                   model(chunk_input_ids, chunk_segment_ids, chunk_input_mask,
                                         prev_hidden_states, chunk_stop_flags,
                                         chunk_start_positions, chunk_end_positions)
                chunk_stop_logits = chunk_stop_logits.detach()
                chunk_stop_probs = F.softmax(chunk_stop_logits, dim=1)
                chunk_stop_probs = chunk_stop_probs[:, 1]
                stop_probs.append(chunk_stop_probs.tolist())
                chunk_stop_logits = chunk_stop_logits.tolist()
                
                if stop_loss is None:
                    stop_loss = chunk_stop_loss
                else:
                    stop_loss += chunk_stop_loss

                if answer_loss is None:
                    answer_loss = chunk_answer_loss
                else:
                    answer_loss += chunk_answer_loss

                if args.supervised_pretraining:
                    chunk_strides = [args.doc_stride] * batch_size
                else:
                    # take movement action
                    chunk_strides = [stride_action_space[stride_ind] for stride_ind in chunk_stride_inds.tolist()]
                cur_global_pointers = [cur_global_pointers[ind] + chunk_strides[ind] for ind in range(len(cur_global_pointers))]
                # put pointer to 0 or the last doc token
                cur_global_pointers = [min(max(0, cur_global_pointers[ind]), len(batch_doc_tokens[ind])-1) \
                                       for ind in range(len(cur_global_pointers))]

                if not args.supervised_pretraining:
                    # reward estimation for reinforcement learning
                    chunk_start_probs = F.softmax(chunk_start_logits.detach(), dim=1).tolist()
                    chunk_end_probs = F.softmax(chunk_end_logits.detach(), dim=1).tolist()
                    #chunk_yes_no_flag_probs = F.softmax(chunk_yes_no_flag_logits.detach(), dim=1).tolist()
                    #chunk_yes_no_ans_probs = F.softmax(chunk_yes_no_ans_logits.detach(), dim=1).tolist()
                    # rewards if stop at the current chunk
                    chunk_stop_rewards = reward_estimation_for_stop(chunk_start_probs, chunk_end_probs,
                                                                    chunk_start_positions.tolist(), chunk_end_positions.tolist(),
                                                                    chunk_stop_flags.tolist())
                    stop_rewards.append(chunk_stop_rewards)

                    # save history (exclude the prob of the last read since the last action is not evaluated)
                    if (t < args.max_read_times - 1):
                        stride_log_probs.append(chunk_stride_log_probs)

            if args.supervised_pretraining:
                loss = (stop_loss * args.stop_loss_weight + answer_loss) / args.max_read_times
            else:
                # stride_log_probs: (bsz, max_read_times-1)
                stride_log_probs = torch.stack(stride_log_probs).transpose(1,0)
                # q_vals: (bsz, max_read_times-1)
                q_vals = reward_estimation(stop_rewards, stop_probs)
                q_vals = torch.tensor(q_vals, dtype=stride_log_probs.dtype, device=device)
                #logger.info("q_vals: {}".format(q_vals))
                reinforce_loss = torch.sum(-stride_log_probs * q_vals, dim=1)
                reinforce_loss = torch.mean(reinforce_loss, dim=0)

                loss = (stop_loss * args.stop_loss_weight + answer_loss) / args.max_read_times + reinforce_loss
            # compute gradients
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # logging training loss
            training_loss += loss.item()
            if (step % 500 == 499):
                #log.write('step: {}, train loss: {}\n'.format(step, training_loss / 500.0))
                logger.info('step: {}, train loss: {}\n'.format(step, training_loss / 500.0))
                if not args.supervised_pretraining:
                    #log.write('q_vals: {}\n'.format(q_vals))
                    logger.info('q_vals: {}\n'.format(q_vals))
                training_loss = 0.0

            # validation on dev data
            if args.do_validate and step % 499 == 0:
                model.eval()
                validate_model(args, model, tokenizer, dev_examples, dev_features,
                               dev_dataloader, dev_ground_truth, device)
                model.train()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        epoch += 1
    

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)


def test_model(args, model, tokenizer, test_examples, test_features, device):
    test_indices = torch.arange(len(test_features), dtype=torch.long)
    test_sampler = SequentialSampler(test_indices)
    test_dataloader = DataLoader(test_indices, sampler=test_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    for step, batch_test_indices in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        batch_test_features = [test_features[ind] for ind in batch_test_indices]
        batch_query_tokens = [f.query_tokens for f in batch_test_features]
        batch_doc_tokens = [f.doc_tokens for f in batch_test_features]
        batch_size = len(batch_test_features)
        cur_global_pointers = [0] * batch_size
        batch_max_doc_length = [args.max_seq_length-3-len(query_tokens) for query_tokens in batch_query_tokens]
        stop_probs = []
        prev_hidden_states = None
        for t in range(args.max_read_times):
            chunk_input_ids, chunk_input_mask, chunk_segment_ids, id_to_tok_maps, _, _, _ = \
                             gen_model_features(cur_global_pointers, batch_query_tokens, batch_doc_tokens, \
                                                None, None, batch_max_doc_length, args.max_seq_length, \
                                                tokenizer, is_train=False)
            chunk_input_ids = torch.tensor(chunk_input_ids, dtype=torch.long, device=device)
            chunk_input_mask = torch.tensor(chunk_input_mask, dtype=torch.long, device=device)
            chunk_segment_ids = torch.tensor(chunk_segment_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                chunk_stop_logits, chunk_stride_inds, chunk_stride_log_probs, \
                                   chunk_start_logits, chunk_end_logits, prev_hidden_states \
                                   = model(chunk_input_ids, chunk_segment_ids, chunk_input_mask, prev_hidden_states)
            # stop_probs: current chunk contains answer
            chunk_stop_logits = chunk_stop_logits.detach().cpu().tolist()
            #chunk_stop_probs = chunk_stop_probs.detach().cpu().tolist()
            #stop_probs.append(chunk_stop_probs[:])

            # find top answer texts for the current chunk
            for i, example_index in enumerate(batch_eval_indices):
                stop_logits = chunk_stop_logits[i]
                start_logits = chunk_start_logits[i].detach().cpu().tolist()
                end_logits = chunk_end_logits[i].detach().cpu().tolist()
                #yes_no_flag_logits = chunk_yes_no_flag_logits[i].detach().cpu().tolist()
                #yes_no_ans_logits = chunk_yes_no_ans_logits[i].detach().cpu().tolist()
                id_to_tok_map = id_to_tok_maps[i]
                example_index = example_index.item()
                all_results.append(RawResult(example_index=example_index,
                                             stop_logits=stop_logits,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             id_to_tok_map=id_to_tok_map))

            # take movement action
            if args.supervised_pretraining:
                chunk_strides = [args.doc_stride] * batch_size
            else:
                chunk_strides = [stride_action_space[stride_ind] for stride_ind in chunk_stride_inds.tolist()]
            cur_global_pointers = [cur_global_pointers[ind] + chunk_strides[ind] for ind in range(len(cur_global_pointers))]
            # put pointer be put to 0 or the last doc token is it
            cur_global_pointers = [min(max(0, cur_global_pointers[ind]), len(batch_doc_tokens[ind])-1) \
                                   for ind in range(len(cur_global_pointers))]

    # write predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    write_predictions(test_examples, test_features, all_results, args.n_best_size, \
                      args.max_answer_length, args.do_lower_case, \
                      output_prediction_file, output_nbest_file, args.verbose_logging)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="Pretrained basic Bert model")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="triviaqa train file")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="triviaqa dev or test file in SQuAD format")
    parser.add_argument("--predict_data_file", default=None, type=str,
                        help="triviaqa dev or test file in Triviaqa format")
    # history queries parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_validate", action='store_true', help="Whether to run validation when training")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    # supervised & reinforcement learning
    parser.add_argument("--supervised_pretraining", action='store_true', help="Whether to do supervised pretraining.")
    #parser.add_argument("--reinforcement_training", action='store_true', help="Whether to do reinforcement learning.")
    #parser.add_argument("--reload_model", action='store_true', help="Load pretrained model for tuning.")
    parser.add_argument("--reload_model_path", type=str, help="Path of pretrained model.")
    parser.add_argument("--recur_type", type=str, default="gated", help="Recurrence model type.")
    # model parameters
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_read_times", default=6, type=int, help="Maximum read times of one document")
    parser.add_argument("--stop_loss_weight", default=1.0, type=float, help="The weight of stop_loss in training")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=2, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    if args.reload_model_path is not None and os.path.isfile(args.reload_model_path):
        logger.info("Reloading pretrained model from {}".format(args.reload_model_path))
        model_state_dict = torch.load(args.reload_model_path)
        model = RCMBert.from_pretrained(args.bert_model,
                                        state_dict=model_state_dict,
                                        action_num=len(stride_action_space),
                                        recur_type=args.recur_type,
                                        allow_yes_no=False)
    elif args.pretrained_model_path is not None and os.path.isdir(args.pretrained_model_path):
        logger.info("Reloading a basic model from  {}".format(args.pretrained_model_path))
        model = RCMBert.from_pretrained(args.pretrained_model_path,
                                        action_num=len(stride_action_space),
                                        recur_type=args.recur_type,
                                        allow_yes_no=False)
    else:
        logger.info("Training a new model from scratch")
        model = RCMBert.from_pretrained(args.bert_model,
                                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                        action_num=len(stride_action_space),
                                        recur_type=args.recur_type,
                                        allow_yes_no=False)
        
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay += ['recur_network', 'stop_network', 'move_stride_network']
    logger.info("Parameter without decay: {}".format(no_decay))
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    """
    examples & features
    """
    train_examples = None
    dev_examples = None
    num_train_steps = None
    if args.do_train:
        cached_train_examples_file = args.train_file+'_train_examples'
        cached_dev_examples_file = args.train_file+'_dev_examples'
        try:
            with open(cached_train_examples_file, "rb") as reader:
                train_examples = pickle.load(reader)
            with open(cached_dev_examples_file, "rb") as reader:
                dev_examples = pickle.load(reader)
            logger.info("Loading train and dev examples...")
        except:
            all_train_examples = read_trivia_examples(
                input_file=args.train_file,
                is_training=True)
            train_examples, dev_examples = split_train_dev_data(all_train_examples)
            with open(cached_train_examples_file, "wb") as writer:
                pickle.dump(train_examples, writer)
            with open(cached_dev_examples_file, "wb") as writer:
                pickle.dump(dev_examples, writer)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        
    
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_RCM_train'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), \
            str(args.max_query_length))
        cached_dev_features_file = args.train_file+'_{0}_{1}_{2}_RCM_dev'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), \
            str(args.max_query_length))
        train_features = None
        dev_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
            with open(cached_dev_features_file, "rb") as reader:
                dev_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_query_length=args.max_query_length,
                is_training=True)
            dev_features = convert_examples_to_features(
                examples=dev_examples,
                tokenizer=tokenizer,
                max_query_length=args.max_query_length,
                is_training=True)
                
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                logger.info("  Saving dev features into cached file %s", cached_dev_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
                with open(cached_dev_features_file, "wb") as writer:
                    pickle.dump(dev_features, writer)
        
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        dev_ground_truth = None
        if args.do_validate and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            logger.info("***** Dev data *****")
            logger.info("  Num orig dev examples = %d", len(dev_examples))
            logger.info("  Num split dev examples = %d", len(dev_features))
            logger.info("  Batch size = %d", args.predict_batch_size)
            # ground truth
            dev_json = json_utils.read_trivia_data(args.predict_data_file)
            dev_ground_truth = trivia_dataset_utils.get_key_to_ground_truth(dev_json)

        train_model(args, model, tokenizer, optimizer, train_examples, train_features,
                    dev_examples, dev_features, dev_ground_truth, device, n_gpu, t_total)



    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # load model
        output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = RCMBert.from_pretrained(args.bert_model,
                                        state_dict=model_state_dict,
                                        action_num=len(stride_action_space),
                                        recur_type=args.recur_type,
                                        allow_yes_no=False)
        model.to(device)
        
        # load data
        test_examples = read_trivia_examples(
            input_file=args.predict_file,
            is_training=False)
        cached_test_features_file = args.train_file+'_{0}_{1}_{2}_RCM_test'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), \
            str(args.max_query_length))
        test_features = None
        try:
            with open(cached_test_features_file, "rb") as reader:
                test_features = pickle.load(reader)
        except:
            test_features = convert_examples_to_features(
                examples=test_examples,
                tokenizer=tokenizer,
                max_query_length=args.max_query_length,
                is_training=False)
            with open(cached_test_features_file, "wb") as writer:
                pickle.dump(test_features, writer)
            
        logger.info("***** Prediction data *****")
        logger.info("  Num test orig examples = %d", len(test_examples))
        logger.info("  Num test split examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.predict_batch_size)
        test_model(args, model, tokenizer, test_examples, test_features, device)
        



if __name__ == "__main__":
    main()




            











