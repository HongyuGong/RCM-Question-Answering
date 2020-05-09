"""
BERT baseline model for CoQA
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
import torch.nn.functional as F
from transformers.tokenization_bert import whitespace_tokenize, BasicTokenizer, BertTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from optimization import BertAdam, warmup_linear
from model.modeling_BERT import BertQA
from data_helper.qa_util import split_train_dev_data
from data_helper.data_helper_coqa import read_coqa_examples
from data_helper.chunk_helper_coqa import convert_examples_to_features, RawResult, \
     make_predictions
from eval_helper.eval_coqa import CoQAEvaluator


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def validate_model(args, model, tokenizer, dev_examples, dev_features,
                   dev_dataloader, dev_evaluator, best_dev_score, device):
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in tqdm(dev_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, batch_yes_no_flag_logits, batch_yes_no_ans_logits \
                                = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            yes_no_flag_logits = batch_yes_no_flag_logits[i].detach().cpu().tolist()
            yes_no_ans_logits = batch_yes_no_ans_logits[i].detach().cpu().tolist()
            dev_feature = dev_features[example_index.item()]
            unique_id = int(dev_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         yes_no_flag_logits=yes_no_flag_logits,
                                         yes_no_ans_logits=yes_no_ans_logits))
    dev_predictions = make_predictions(dev_examples, dev_features, all_results,
                                        args.n_best_size, args.max_answer_length, args.do_lower_case,
                                        args.verbose_logging, validate_flag=True)
    dev_scores = dev_evaluator.model_performance(dev_predictions)
    dev_score = dev_scores['overall']['f1']
    logger.info('dev score: {}'.format(dev_score))
    if dev_score > best_dev_score:
        best_model_to_save = model.module if hasattr(model, 'module') else model
        best_output_model_file = os.path.join(args.output_dir, "best_BERT_model.bin")
        torch.save(best_model_to_save.state_dict(), best_output_model_file)
        best_dev_score = max(best_dev_score, dev_score)
        logger.info("Best dev score: {}, saved to best_BERT_model.bin".format(dev_score))
    return best_dev_score


def train_model(args, model, tokenizer, optimizer, train_examples, train_features,
                dev_examples, dev_features, dev_evaluator, device, n_gpu, t_total):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    all_yes_no_flags = torch.tensor([f.yes_no_flag for f in train_features], dtype=torch.long)
    all_yes_no_answers = torch.tensor([f.yes_no_ans for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions, all_yes_no_flags, all_yes_no_answers)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
  
    #if args.do_validate and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    if args.do_validate:
        all_dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_dev_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        all_dev_example_index=torch.arange(all_dev_input_ids.size(0), dtype=torch.long)
        dev_data = TensorDataset(all_dev_input_ids, all_dev_input_mask,
                                 all_dev_segment_ids, all_dev_example_index)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler,
                                     batch_size=args.predict_batch_size)
        
    # ****************** Train & Validate ******************
    best_dev_score = 0.0
    epoch = 0
    global_step = 0
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        training_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions, \
                       yes_no_flags, yes_no_answers = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, \
                         yes_no_flags, yes_no_answers)
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
                logger.info('step {}, avg loss: {}\n'.format(step, training_loss / 500.0))
                training_loss = 0.0

            # validation
            #if (epoch >=1 and step % 500 == 499):
            if args.do_validate and step % 500 == 499:
                model.eval()
                best_dev_score = validate_model(args, model, tokenizer, dev_examples, dev_features,
                                                dev_dataloader, dev_evaluator, best_dev_score, device)
                model.train()

            # change learning rate
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
    output_model_file = os.path.join(args.output_dir, "BERT_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
        

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="triviaqa train file")
    parser.add_argument("--use_history", default=False, action="store_true")
    parser.add_argument("--append_history", default=False, action="store_true", help="Whether to append the previous queries to the current one.")
    parser.add_argument("--n_history", default=-1, type=int, help="The number of previous queries used in current query.")
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
    # model parameters
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=60, type=int,
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

    if not args.do_train:
        raise ValueError("`do_train` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.do_lower_case)

    logger.info("Training BERT model")
    model = BertQA.from_pretrained(args.bert_model,
                                   cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                   allow_yes_no=True)
    
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
    logger.info("Parameter without decay: {}".format(no_decay))
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # CoQA Example and ChunkFeature
    train_examples = None
    num_train_steps = None
    if args.do_train:
        # Load examples
        cached_train_examples_file = args.train_file+'_train_examples'
        cached_dev_examples_file = args.train_file+'_dev_examples'
        try:
            with open(cached_train_examples_file, "rb") as reader:
                train_examples = pickle.load(reader)
            with open(cached_dev_examples_file, "rb") as reader:
                dev_examples = pickle.load(reader)
            logger.info("Loading train and dev examples...")
        except:
            all_train_examples = read_coqa_examples(
                input_file=args.train_file,
                is_training=True,
                use_history=args.use_history,
                n_history=args.n_history)
            train_examples, dev_examples = split_train_dev_data(all_train_examples)
            with open(cached_train_examples_file, "wb") as writer:
                pickle.dump(train_examples, writer)
            with open(cached_dev_examples_file, "wb") as writer:
                pickle.dump(dev_examples, writer)
            logger.info("Creating train and dev examples...")
        logger.info("# of train examples: {}, # of dev examples: {}".format(
            len(train_examples), len(dev_examples)))

        # Load features
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_BERT_train'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), \
            str(args.max_query_length))
        cached_dev_features_file = args.train_file+'_{0}_{1}_{2}_BERT_dev'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), \
            str(args.max_query_length))
        train_features = None
        dev_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
            with open(cached_dev_features_file, "rb") as reader:
                dev_features = pickle.load(reader)
            print("Done loading features...")
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                append_history=args.append_history)
            dev_features = convert_examples_to_features(
                examples=dev_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                append_history=args.append_history)
            print("Done creating features...")
            
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                logger.info("  Saving dev features into cached file %s", cached_dev_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
                with open(cached_dev_features_file, "wb") as writer:
                    pickle.dump(dev_features, writer)

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

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
        logger.info("***** Running training *****")
        logger.info("  Num train orig examples = %d", len(train_examples))
        logger.info("  Num train split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.do_validate and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            logger.info("***** Dev data *****")
            logger.info("  Num orig dev examples = %d", len(dev_examples))
            logger.info("  Num split dev examples = %d", len(dev_features))
            logger.info("  Batch size = %d", args.predict_batch_size)
            dev_evaluator = CoQAEvaluator(dev_examples)

        train_model(args, model, tokenizer, optimizer, train_examples, train_features,
                    dev_examples, dev_features, dev_evaluator, device, n_gpu, t_total)
        

if __name__ == "__main__":
    main()




