"""
preprocess CoQA for extractive question answering
"""

import argparse
import json
import re
import time
import string
from collections import Counter
from collections import OrderedDict
from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:9000')
UNK = 'unknown'


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def process(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json',
                             'ssplit.newlineIsSentenceBreak': 'two'})

    output = {'word': [], 'offsets': []}

    try:
        for sent in paragraph['sentences']:
            for token in sent['tokens']:
                output['word'].append(_str(token['word']))
                output['offsets'].append((token['characterOffsetBegin'], token['characterOffsetEnd']))
    except:
        print("error in line 45: {}".format(paragraph))
    return output


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


def remove_last_punc(s):
    exclude = set(string.punctuation)
    for start_pos in range(len(s)):
        ch = s[start_pos]
        if ch not in exclude:
            break
    for end_pos in reversed(range(len(s))):
        ch = s[end_pos]
        if (ch not in exclude):
            break
    new_s = s[start_pos:end_pos+1]
    remove_flag = (len(new_s) < len(s))
    """
    if remove_flag:
        print("s: {}\t removed s: {}".format(s, new_s))
    """
    return new_s, int(remove_flag)


def find_span_with_gt(context, offsets, ground_truth):
    """
    find answer from paragraph so that best f1 score is achieved
    """
    best_f1 = 0.0
    best_span = (len(context)-1, len(context)-1)
    gt = normalize_answer(ground_truth).split()

    ls = [i for i in range(len(offsets))
          if context[offsets[i][0]:offsets[i][1]].lower() in gt]

    for i in range(len(ls)):
        for j in range(i, len(ls)):
            pred = normalize_answer(context[offsets[ls[i]][0]: offsets[ls[j]][1]]).split()
            common = Counter(pred) & Counter(gt)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(gt)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_span = (offsets[ls[i]][0], offsets[ls[j]][1])
    return best_span, best_f1
    
    
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    fscore_threshold=0.6

    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    data = OrderedDict()
    data['version'] = dataset['version']
    _datum_list = []
    yes_cnt = 0
    no_cnt = 0
    inaccurate_cnt = 0
    total_rpunc_cnt = 0
    for i, datum in enumerate(dataset['data']):
        if (i % 500 == 0):
            print("processing {} examples".format(i))
            
        _datum = OrderedDict()
        source = datum["source"]
        paragraph_id = datum["id"]
        filename = datum["filename"]
        context_str = datum["story"]
        _datum["source"] = source
        _datum["id"] = paragraph_id
        _datum["filename"] = filename
        _datum["story"] = context_str
        _datum["story"] += " "
        _datum["story"] += UNK

        annotated_context_str = process(context_str)
        offsets = annotated_context_str['offsets']
        offsets.append((len(_datum['story']) - len(UNK), len(_datum['story'])))

        # ?? add additional_answers ??
        if ('additional_answers' in datum):
            _datum['additional_answers'] = datum['additional_answers']

        # question & answer
        questions = []
        answers = []
        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            turn_id = question['turn_id']
            question_text = question['input_text']
            cur_question = OrderedDict()
            cur_question['input_text'] = question_text
            cur_question['turn_id'] = turn_id
            questions.append(cur_question)
            
            ans_span_text = answer['span_text']
            ans_input_text = answer['input_text']
            ans_start = answer['span_start']
            ans_end = answer['span_end']
            cur_answer = OrderedDict() # attr: span_start, span_end, text, turn_id
            start = ans_start
            end = ans_end
            chosen_text = _datum['story'][ans_start:ans_end].lower()
            # remove space from start
            while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                chosen_text = chosen_text[1:]
                start += 1
            # remove space from end
            while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                chosen_text = chosen_text[:-1]
                end -= 1

            input_text = ans_input_text.strip().lower()

            # remove punc from input_text
            input_text, remove_cnt = remove_last_punc(input_text)
            total_rpunc_cnt += remove_cnt
            
            # yes_no_flag, yes_no_ans
            # deal with yes/no question
            if (input_text == 'yes' or input_text == 'yes.' or input_text ==".yes"):
                yes_no_flag = 1
                yes_no_ans = 1
                yes_cnt += 1
                input_text = 'yes'
            elif (input_text == 'no' or input_text == 'no.' or input_text == ".no"):
                yes_no_flag = 1
                yes_no_ans = 0
                no_cnt += 1
                input_text = 'no'
            else:
                yes_no_flag = 0
                yes_no_ans = -1

            if (input_text in ["yes", "no"]):
                ans_span = (start, end)
            elif input_text in chosen_text:
                i = chosen_text.find(input_text)
                ans_span = (start + i, start + i + len(input_text))
            else:
                ans_span, fscore = find_span_with_gt(_datum['story'], offsets, input_text)
                """
                if input_text == 'unknown':
                    print("input_text: {}, chosen_text: {}, fscore: {}, ans_span".format(input_text, \
                                                                               chosen_text, fscore))
                """
                if (fscore < fscore_threshold):
                    ans_span = (start, end)
                    inaccurate_cnt += 1
            
            cur_answer['span_start'] = ans_span[0]
            cur_answer['span_end'] = ans_span[1]
            cur_answer['yes_no_flag'] = yes_no_flag
            cur_answer['yes_no_ans'] = yes_no_ans
            cur_answer['text'] = _datum['story'][ans_span[0] : ans_span[1]]
            cur_answer['input_text'] = answer['input_text']
            cur_answer['span_text'] = answer['span_text']
            cur_answer['turn_id'] = answer['turn_id']
            answers.append(cur_answer)
        _datum["questions"] = questions[:]
        _datum["answers"] = answers[:]
        _datum_list.append(_datum)
    data['data'] = _datum_list
    print("# of answer yes: {}, # of answer no: {}".format(yes_cnt, no_cnt))
    print("inaccurate cnt: {}".format(inaccurate_cnt))
    print("total_rpunc_cnt: {}".format(total_rpunc_cnt))
    
    with open(args.output_file, 'w') as output_file:
        json.dump(data, output_file, sort_keys=True, indent=4)



