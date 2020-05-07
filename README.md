# RCM-Question-Answering

# This repository is being constructed and tested. Training and testing instructions will be released after testing.

This is the re-implementation of RCM-BERT model for question answering in the paper "Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension".

Required package:

(1) python3;

(2) torch;

(3) [Transformers by Huggingface](https://github.com/huggingface/transformers).


## Data Preparation

1. CoQA

(1) Download data from [CoQA website](https://stanfordnlp.github.io/coqa/) and save to DATA_DIR. 

Enter the directory RCM-Question-Answering/src/.

(2) Preprocess CoQA data

ref: https://github.com/stanfordnlp/coqa-baselines/blob/master/README.md

Start a CoreNLP server

```bash
  mkdir lib
  wget -O lib/stanford-corenlp-3.9.1.jar https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/3.9.1/stanford-corenlp-3.9.1.jar
  java -mx4g -cp lib/stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

Run a script to preprocess data 

```bash
python3 data_helper/preprocess_coqa.py --data_file DATA_DIR/coqa-train-v1.0.json --output_file DATA_DIR/coqa.train.json
python3 data_helper/preprocess_coqa.py --data_file DATA_DIR/coqa-dev-v1.0.json --output_file DATA_DIR/coqa.dev.json
```

2. QuAC

Download data from [QuAC websiet](https://quac.ai)

3. TriviaQA

Download data from [TriviaQA websiet](https://nlp.cs.washington.edu/triviaqa/). Two folders qa/ and evidence/ can be found in triviaqa/.

Follow instructions [here](https://github.com/mandarjoshi90/triviaqa) to adapt TriviaQA to SQuAD format. Create a subfolder under squad-qa/ under the folder triviaqa/.


## Training

For the efficiency of model training, we try first-pretrain-then-train approach. The model is first pre-trained with fixed strides and no recurrence. Then the recurrent chunking mechanism is applied to further train the model to chunk documents with flexible strides and progapagte informaiton among segmenets with recurrence.

### 1. Conversational Question Answering (CoQA)

#### Pretrain CoQA model

```bash
python3 train/run_BERT_coqa.py
--bert_model bert-large-uncased
--output_dir OUTPUT_DIR/pretrained/
--train_file DATA_DIR/coqa.train.josn
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH[256]
--doc_stride 64
--max_query_length 64
--do_train
--do_validate
--train_batch_size 12
--predict_batch_size 18
--learning_rate 3e-5
--num_train_epochs 2.5
--max_answer_length 40
--do_lower_case
```

#### Recurrent chunking mechamism (RCM) for CoQA.

```bash
python3 train/run_RCM_coqa.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--train_file DATA_DIR/coqa.train.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride 64
--do_train
--do_validate
--do_lower_case
--pretrained_model_path OUTPUT_DIR/pretrained/best_RCM_model.bin
--recur_type RECUR_TYPE
--train_batch_size 8
--learning_rate 1e-5
--num_train_epochs 2.0
--max_read_times 3
--max_answer_length 40
```

* MAX_SEQ_LENGTH can be integers no larger than 512

* RECUR_TYPE can be "gated" or "lstm"

* OUTPUD_DIR: the path where the trained model will be saved


#### Prediction

```bash
python3 train/run_RCM_coqa.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--predict_file DATA_DIR/coqa.dev.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride DOC_STRIDE
--do_predict
--do_lower_case
--recur_type RECUR_TYPE
--predict_batch_size 12
--max_read_times 3
--max_answer_length 40
```

* Predictions will be saved in OUTPUT_DIR/rl/


### 2. Question Answering in Context (QuAC)

#### Pretrain QuAC model

```bash
python3 train/run_BERT_quac.py
--bert_model bert-large-uncased
--output_dir OUTPUT_DIR/pretrained/
--train_file DATA_DIR/train_v0.2.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH[256]
--doc_stride 64
--max_query_length 64
--do_train
--do_validate
--train_batch_size 12
--predict_batch_size 18
--learning_rate 3e-5
--num_train_epochs 2.5
--max_answer_length 40
--do_lower_case
```

#### Recurrent chunking mechamism (RCM) for QuAC

```bash
python3 train/run_RCM_quac.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--train_file DATA_DIR/train_v0.2.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride 64
--do_train
--do_validate
--do_lower_case
--pretrained_model_path OUTPUT_DIR/pretrained/best_RCM_model.bin
--recur_type RECUR_TYPE
--train_batch_size 8
--learning_rate 1e-5
--num_train_epochs 2.0
--max_read_times 3
--max_answer_length 40
```

* MAX_SEQ_LENGTH can be integers no larger than 512

* RECUR_TYPE can be "gated" or "lstm"

* OUTPUD_DIR: the path where the trained model will be saved


#### Prediction

```bash
python3 train/run_RCM_quac.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--predict_file DATA_DIR/val_v0.2.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride DOC_STRIDE
--do_predict
--do_lower_case
--recur_type RECUR_TYPE
--predict_batch_size 12
--max_read_times 3
--max_answer_length 40
```

* Predictions will be saved in OUTPUT_DIR/rl/


### 3. TriviaQA

#### Pretrain Trivia model

```bash
python3 train/run_BERT_trivia.py
--bert_model bert-large-uncased
--output_dir OUTPUT_DIR/pretrained/
--train_file DATA_DIR/trivia.train.josn
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH[256]
--doc_stride 64
--max_query_length 64
--do_train
--do_validate
--train_batch_size 12
--predict_batch_size 18
--learning_rate 3e-5
--num_train_epochs 2.5
--max_answer_length 40
--do_lower_case
```

#### Recurrent chunking mechamism (RCM) for Trivia

```bash
python3 train/run_RCM_trivia.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--train_file DATA_DIR/trivia.train.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride 64
--do_train
--do_validate
--do_lower_case
--pretrained_model_path OUTPUT_DIR/pretrained/best_RCM_model.bin
--recur_type RECUR_TYPE
--train_batch_size 8
--learning_rate 1e-5
--num_train_epochs 2.0
--max_read_times 3
--max_answer_length 40
```

* MAX_SEQ_LENGTH can be integers no larger than 512

* RECUR_TYPE can be "gated" or "lstm"

* OUTPUD_DIR: the path where the trained model will be saved


#### Prediction

```bash
python3 train/run_RCM_trivia.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/rl/
--predict_file DATA_DIR/trivia.dev.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride DOC_STRIDE
--do_predict
--do_lower_case
--recur_type RECUR_TYPE
--predict_batch_size 12
--max_read_times 3
--max_answer_length 40
```

* Predictions will be saved in OUTPUT_DIR/rl/


## Ablation Study

Train CoQA model with recurrence but without flexible strides

```bash
python3 train/run_RCM_coqa.py 
--bert_model bert-large-uncased 
--output_dir OUTPUT_DIR/pretrained/
--train_file DATA_DIR/coqa.train.json
--use_history
--n_history -1
--max_seq_length MAX_SEQ_LENGTH
--max_query_length 64
--doc_stride 64
--do_train
--do_validate
--do_lower_case
--recur_type RECUR_TYPE
--supervised_pretraining
--train_batch_size 8
--learning_rate 3e-5
--num_train_epochs 2.0
--max_read_times 3
--max_answer_length 40
```

If you have any questions, please contact Hongyu Gong (hgong6@illinois.edu).

If you use our code, please cite our work:
Hongyu Gong, Yelong Shen, Dian Yu, Jianshu Chen and Dong Yu, "Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension", accepted by Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020).
