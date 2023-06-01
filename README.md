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
  - [SIP+Multi-label classification](#Multi-label-classification) 
  - [SIP+Sequence generation](#Sequence-generation)
- [Evaluation](#Evaluation) 
  - [Evaluate SIP](#Evaluate-SIP) 
  - [Evaluate action prediction](#Evaluate-Action-prediction)

Note that due to the nature of the anonymous repository, we apologize that the outline hyperlinks do not work. 
However, these issues will be fixed in the final repository.

## Prerequisites
Install dependencies:  
```bash
pip install -r requirements.txt
```

## Data Preprocessing
The following commands are used to conduct data preprocessing, which also includes the automatic annotation of initiative-taking decision labels. 
We derive the initiative annotations by mapping the manual annotations of actions to initiative or non-initiative labels
The raw data for the WISE, MSDialog and ClariQ is stored in the paths `./data/WISE/`, `./data/MSDialog/` and `./data/ClariQ/`.
Note that the preprocessed data is still stored in these paths.

Run the following commands to preprocess WISE:
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
Run the following commands to preprocess MSDialog:
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
Run the following commands to preprocess ClariQ:
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

## SIP

### LLaMA
```bash
```

### MuSIc
```bash
```

## Clarification need prediction
```bash
```

## Action prediction
### Multi-label classification
```bash
```

### Sequence generation
```bash
```

### SIP+multi-label classification
```bash
```

### SIP+sequence generation
```bash
```


## Evaluation

### Evaluate SIP
```bash
```

### Evaluate action prediction
```bash
```