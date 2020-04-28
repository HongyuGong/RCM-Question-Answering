import json, string, re
from collections import Counter, defaultdict
from argparse import ArgumentParser

MIN_F1 = 0.4

class QuACEvaluator():

    def __init__(self, examples):
        self.gold_data = QuACEvaluator.gold_answers_to_dict(examples)

    @staticmethod
    def gold_answers_to_dict(example):
        self.gold_data = defaultdict(dict)
        for example in examples:
            qid = example.example_id
            dia_id = qid.split("_q#")[0]
            ids = example_id.split("_q#")
            self.gold_data[dia_id][qid] = example

    def eval_fn(self, pred_data):
        span_overlap_stats = Counter()
        sentence_overlap = 0.
        para_overlap = 0.
        total_qs = 0.
        f1_stats = defaultdict(list)
        unfiltered_f1s = []
        human_f1 = []
        HEQ = 0.
        DHEQ = 0.
        total_dials = 0.
        #yes_nos = []
        #followups = []
        unanswerables = []
        for dia_id in self.gold_data:
            for qid in self.gold_data[dia_id]:
                example = self.gold_data[dia_id][qid]
                val_spans = [example.orig_answer_text]
                val_spans = handle_cannot(val_spans)
                hf1 = leave_one_out(val_spans)

                if dia_id not in pred_data or qid not in pred_data[dia_id]:
                    print(dia_id, qid, 'no prediction for this dialogue id')
                    good_dial = 0
                    f1_stats['NO ANSWER'].append(0.0)
                    if val_spans == ['CANNOTANSWER']:
                        unanswerables.append(0.0)
                    total_qs += 1
                    unfiltered_f1s.append(0.0)
                    if hf1 >= MIN_F1:
                        human_f1.append(hf1)
                    continue
                
                pred_span = model_results[dia_id][qid]
                context = " ".join(example.doc_tokens)
                max_overlap, _ = metric_max_over_ground_truths( \
                    pred_span, val_spans, context)
                max_f1 = leave_one_out_max( \
                    pred_span, val_spans, context)
                unfiltered_f1s.append(max_f1)
                
                # dont eval on low agreement instances
                if hf1 < MIN_F1:
                    continue

                human_f1.append(hf1)
                if val_spans == ['CANNOTANSWER']:
                    unanswerables.append(max_f1)
                if verbose:
                    print("-" * 20)
                    print(pred_span)
                    print(val_spans)
                    print(max_f1)
                    print("-" * 20)
                if max_f1 >= hf1:
                    HEQ += 1.
                else:
                    good_dial = 0.
                span_overlap_stats[max_overlap] += 1
                f1_stats[max_overlap].append(max_f1)
                total_qs += 1.
            DHEQ += good_dial
            total_dials += 1
    DHEQ_score = 100.0 * DHEQ / total_dials
    HEQ_score = 100.0 * HEQ / total_qs
    all_f1s = sum(f1_stats.values(), [])
    overall_f1 = 100.0 * sum(all_f1s) / len(all_f1s)
    unfiltered_f1 = 100.0 * sum(unfiltered_f1s) / len(unfiltered_f1s)
    #yesno_score = (100.0 * sum(yes_nos) / len(yes_nos))
    #followup_score = (100.0 * sum(followups) / len(followups))
    unanswerable_score = (100.0 * sum(unanswerables) / len(unanswerables))
    metric_json = {"unfiltered_f1": unfiltered_f1, "f1": overall_f1, "HEQ": HEQ_score, \
                   "DHEQ": DHEQ_score, "unanswerable_acc": unanswerable_score}
    if verbose:
        print("=======================")
        display_counter('Overlap Stats', span_overlap_stats, f1_stats)
    print("=======================")
    print('Overall F1: %.1f' % overall_f1)
    print('Unfiltered F1 ({0:d} questions): {1:.1f}'.format(len(unfiltered_f1s), unfiltered_f1))
    print('Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)'.format(unanswerable_score, len(unanswerables)))
    print('Human F1: %.1f' % (100.0 * sum(human_f1) / len(human_f1)))
    print('Model F1 >= Human F1 (Questions): %d / %d, %.1f%%' % (HEQ, total_qs, 100.0 * HEQ / total_qs))
    print('Model F1 >= Human F1 (Dialogs): %d / %d, %.1f%%' % (DHEQ, total_dials, 100.0 * DHEQ / total_dials))
    print("=======================")
    return metric_json    

        
    @staticmethod
    def is_overlapping(x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))

    @staticmethod
    def handle_cannot(refs):
        num_cannot = 0
        num_spans = 0
        for ref in refs:
            if ref == 'CANNOTANSWER':
                num_cannot += 1
            else:
                num_spans += 1
        if num_cannot >= num_spans:
            refs = ['CANNOTANSWER']
        else:
            refs = [x for x in refs if x != 'CANNOTANSWER']
        return refs

    @staticmethod
    def leave_one_out(refs):
        if len(refs) == 1:
            return 1.
        splits = []
        for r in refs:
            splits.append(r.split())
        t_f1 = 0.0
        for i in range(len(refs)):
            m_f1 = 0
            for j in range(len(refs)):
                if i == j:
                    continue
                f1_ij = f1_score(refs[i], refs[j])
                if f1_ij > m_f1:
                    m_f1 = f1_ij
            t_f1 += m_f
        return t_f1 / len(refs)

    @staticmethod
    def leave_one_out_max(prediction, ground_truths, article):
        if len(ground_truths) == 1:
            return metric_max_over_ground_truths(prediction, ground_truths, article)[1]
        else:
            t_f1 = []
            # leave out one ref every time
            for i in range(len(ground_truths)):
                idxes = list(range(len(ground_truths)))
                idxes.pop(i)
                refs = [ground_truths[z] for z in idxes]
                t_f1.append(metric_max_over_ground_truths(prediction, refs, article)[1])
        return 1.0 * sum(t_f1) / len(t_f1)

    @staticmethod
    def compute_span_overlap(pred_span, gt_span, text):
        if gt_span == 'CANNOTANSWER':
            if pred_span == 'CANNOTANSWER':
                return 'Exact match', 1.0
            return 'No overlap', 0.
        fscore = f1_score(pred_span, gt_span)
        pred_start = text.find(pred_span)
        gt_start = text.find(gt_span)

        if pred_start == -1 or gt_start == -1:
            return 'Span indexing error', fscore

        pred_end = pred_start + len(pred_span)
        gt_end = gt_start + len(gt_span)

        fscore = f1_score(pred_span, gt_span)
        overlap = is_overlapping(pred_start, pred_end, gt_start, gt_end)

        if exact_match_score(pred_span, gt_span):
            return 'Exact match', fscore
        if overlap:
            return 'Partial overlap', fscore
        else:
            return 'No overlap', fscore


    @staticmethod
    def metric_max_over_ground_truths(prediction, ground_truths, article):
        cores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = compute_span_overlap(prediction, ground_truth, article)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths, key=lambda x: x[1])

    @staticmethod
    def display_counter(title, c, c2=None):
        print(title)
        for key, _ in c.most_common():
            if c2:
                print('%s: %d / %d, %.1f%%, F1: %.1f' % (
                    key, c[key], sum(c.values()), c[key] * 100. / sum(c.values()), sum(c2[key]) * 100. / len(c2[key])))
            else:
                print('%s: %d / %d, %.1f%%' % (key, c[key], sum(c.values()), c[key] * 100. / sum(c.values())))




