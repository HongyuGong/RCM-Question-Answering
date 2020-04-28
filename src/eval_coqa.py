"""Official evaluation script for CoQA.

The code is based partially on SQuAD 2.0 evaluation script.
"""
import argparse
import json
import re
import string
import sys

from collections import Counter, OrderedDict

#OPTS = None

#out_domain = ["reddit", "science"]
#in_domain = ["mctest", "gutenberg", "race", "cnn", "wikipedia"]
#domain_mappings = {"mctest":"children_stories", "gutenberg":"literature", "race":"mid-high_school", "cnn":"news", "wikipedia":"wikipedia", "science":"science", "reddit":"reddit"}

class CoQAEvaluator():
   
    def __init__(self, examples):
        self.gold_data = CoQAEvaluator.gold_answers_to_dict(examples)

    @staticmethod
    def gold_answers_to_dict(examples):
        gold_dict = {}
        for example in examples:
            story_id = example.paragraph_id
            turn_id = example.turn_id
            gold_answer = example.orig_answer_text
            key = (story_id, turn_id)
            gold_dict[key] = gold_answer
        return gold_dict
            
    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return CoQAEvaluator.normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(CoQAEvaluator.normalize_answer(a_gold) == CoQAEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = CoQAEvaluator.get_tokens(a_gold)
        pred_toks = CoQAEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_turn_score(self, a_gold, a_pred):
        em = CoQAEvaluator.compute_exact(a_gold, a_pred)
        f1 = CoQAEvaluator.compute_f1(a_gold, a_pred)
        return {"em": em, "f1": f1}

    def get_raw_scores(self, pred_data):
        exact_scores = {}
        f1_scores = {}
        for key in self.gold_data:
            if key not in pred_data:
                sys.stderr.write('Missing prediction for {} and turn_id: {}\n'.format(story_id, turn_id))
                continue
            a_gold = self.gold_data[key]
            a_pred = pred_data[key]
            scores = self.compute_turn_score(a_gold, a_pred)
            exact_scores[key] = scores["em"]
            f1_scores[key] = scores["f1"]
        return exact_scores, f1_scores


    def model_performance(self, pred_data):
        exact_scores, f1_scores = self.get_raw_scores(pred_data)
        assert len(exact_scores), len(f1_scores)
        turn_count= len(exact_scores)
        em_total = np.sum([exact_scores[key] for key in exact_scores])
        f1_total = np.sum([f1_scores[key] for key in f1_scores])
        scores["overall"] = {'em': round(em_total / max(1, turn_count) * 100, 1),
                             'f1': round(f1_total / max(1, turn_count) * 100, 1),
                             'turns': turn_count}
        return scores
        
