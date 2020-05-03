"""
BERT for question answering on CoQA dataset
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

from model.modeling_BERT import BertQA
from train.optimization import BertAdam, warmup_linear
from data_helper.qa_util import split_train_dev_data, gen_model_features, _improve_answer_span, \
     get_final_text, _compute_softmax, _get_best_indexes
from data_helper_coqa import read_coqa_examples, convert_examples_to_features, \
     RawResult, make_predictions, write_predictions
from eval_helper.eval_coqa import CoQAEvaluator
