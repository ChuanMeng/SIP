# System Initiative Prediction (SIP)

This is the virtual appendix for the paper **System Initiative Prediction for Multi-turn Conversational Information Seeking**.
In order to replicate the results in the paper, kindly adhere to the subsequent steps:
- [Prerequisites](#Prerequisites)
- [Data Preprocessing](#Data-Preprocessing)
- [SIP](#SIP)
  - [LLaMA](#LLaMA) 
  - [MuSIc](#MuSIc) 
- [Clarification need prediction](#Clarification-need-prediction) 
- [Action prediction](#Action-prediction) 
  - [Multi-label classification](#Multi-label-classification) 
  - [Sequence generation](#Sequence-generation)
- [Evaluation](#Evaluation) 
  - [Evaluate SIP](#Evaluate-SIP) 
  - [Evaluate Action prediction](#Evaluate-Action-prediction)


## Prerequisites
Install dependencies:  
```bash
pip install -r requirements.txt
```

## Data Preprocessing
S`./output/pre-retrieval/`. 
```bash
python -u ./dataset/preprocess_WISE.py \
--input_path ./dataset/WISE/conversation_train_line.json \
--output_path ./dataset/WISE/train_WISE.pkl

python -u ./dataset/preprocess_WISE.py \
--input_path ./dataset/WISE/conversation_valid_line.json \
--output_path ./dataset/WISE/valid_WISE.pkl

python -u ./dataset/preprocess_WISE.py \
--input_path ./dataset/WISE/conversation_test_line.json \
--output_path ./dataset/WISE/test_WISE.pkl
```

```bash
python -u ./dataset/preprocess_MSDialog.py \
--input_path ./dataset/MSDialog/train.tsv \
--output_path ./dataset/MSDialog/train_MSDialog.pkl

python -u ./dataset/preprocess_MSDialog.py \
--input_path ./dataset/MSDialog/valid.tsv \
--output_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u ./dataset/preprocess_MSDialog.py \
--input_path ./dataset/MSDialog/test.tsv \
--output_path ./dataset/MSDialog/test_MSDialog.pkl
```


```bash
python -u ./dataset/preprocess_ClariQ.py \
--input_path ./dataset/ClariQ/train.tsv \
--output_path ./dataset/ClariQ/train_ClariQ.pkl

python -u ./dataset/preprocess_ClariQ.py \
--input_path ./dataset/ClariQ/dev.tsv \
--output_path ./dataset/ClariQ/valid_ClariQ.pkl

python -u ./dataset/preprocess_ClariQ.py \
--input_path ./dataset/ClariQ/test_with_labels.tsv \
--output_path ./dataset/ClariQ/test_ClariQ.pkl
```