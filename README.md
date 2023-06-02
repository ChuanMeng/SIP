# System Initiative Prediction (SIP)

This is the virtual appendix for the paper **System Initiative Prediction for Multi-turn Conversational Information Seeking**.
In order to replicate the results in the paper, kindly adhere to the subsequent steps:
- [Prerequisites](#Prerequisites)
- [Data Preprocessing](#Data-Preprocessing)
- [SIP](#SIP)
  - [LLaMA](#LLaMA) 
  - [MuSIc](#MuSIc) 
- [Clarification need prediction](#Clarification-need-prediction) 
- [Evaluate SIP and Clarification need prediction](#Evaluate-SIP-and-clarification-need-prediction) 
- [Action prediction](#Action-prediction) 
  - [Multi-label classification](#Multi-label-classification) 
  - [SIP+Multi-label classification](#Multi-label-classification) 
  - [Sequence generation](#Sequence-generation)
  - [SIP+Sequence generation](#Sequence-generation)
-[Evaluate action prediction](#Evaluate-Action-prediction)

Note that due to the nature of the anonymous repository, we apologize that the outline hyperlinks do not work. 
However, these issues will be fixed in the final repository.

## Prerequisites
Install dependencies:  
```bash
pip install -r requirements.txt
```

## Data Preprocessing
The following commands are used to conduct data preprocessing, which includes the automatic annotation of initiative-taking decision labels. 
We derive the initiative annotations by mapping the manual annotations of actions to initiative or non-initiative labels.
The raw data for the WISE, MSDialog and ClariQ is stored in the paths `./data/WISE/`, `./data/MSDialog/` and `./data/ClariQ/`.
The preprocessed data is still stored in these paths.

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

#### WISE
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/WISE/train_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/WISE/valid_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

#### MSDialog
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/MSDialog/train_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/MSDialog/valid_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

## Clarification need prediction
```bash
srun -p gpu --gres=gpu:a6000:1 --time=99:00:00 -c8 --mem=50G \
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/train_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train \
# --initialization_path ./output/MSDialog.SIP.DistanceCRF/checkpoints/1.pkl
```

```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/valid_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/test_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```
## Evaluate SIP and Clarification need prediction
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.DistanceCRF \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.DistanceCRF \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```

```bash
python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF \
--label_path ./dataset/ClariQ/test_ClariQ.pkl

python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF-TL \
--label_path ./dataset/ClariQ/test_ClariQ.pkl
```

## Action prediction

### Multi-label classification

#### WISE
```bash
python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/WISE/train_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/WISE/valid_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

#### MSDialog
```bash
python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/MSDialog/train_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/MSDialog/valid_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task AP \
--model mlc \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

### SIP+multi-label classification

#### WISE
```bash
python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/WISE/train_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/WISE/valid_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/WISE.SIP.DistanceCRF/3.txt \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/WISE.SIP.DistanceCRF/3.txt \
--mode inference
```

#### MSDialog
```bash
python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/MSDialog/train_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/MSDialog/valid_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/MSDialog.SIP.DistanceCRF/1.txt \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/MSDialog.SIP.DistanceCRF/1.txt \
--mode inference
```

### Sequence generation

#### WISE
```bash
python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/WISE/train_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/WISE/valid_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference

python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

#### MSDialog
```bash
python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/MSDialog/train_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/MSDialog/valid_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode valid

python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```

### SIP+sequence generation
#### WISE
```bash
python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/WISE/train_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/WISE/valid_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/WISE.SIP.DistanceCRF/3.txt \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/WISE.SIP.DistanceCRF/3.txt \
--mode inference
```

#### MSDialog
```bash
python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/MSDialog/train_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/MSDialog/valid_MSDialog \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/MSDialog.SIP.DistanceCRF/1.txt \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path ./output/MSDialog.SIP.DistanceCRF/1.txt \
--mode inference
```

## Evaluate action prediction
```bash

python -u Evaluation.py \
--prediction_path ./output/WISE.AP.mlc \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.mlc \
--label_path ./dataset/MSDialog/test_MSDialog.pkl


python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.mlc \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.mlc \
--label_path ./dataset/MSDialog/test_MSDialog.pkl


python -u Evaluation.py \
--prediction_path ./output/WISE.AP.sg \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.sg \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.sg \
--label_path ./dataset/WISE/test_WISE.pkl


python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.sg \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

```