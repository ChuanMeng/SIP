# System Initiative Prediction (SIP)

This is the virtual appendix for the paper **System Initiative Prediction for Multi-turn Conversational Information Seeking**.
In order to replicate the results in the paper, kindly adhere to the subsequent steps:
- [Prerequisites](#Prerequisites)
- [Data Preprocessing](#Data-Preprocessing)
- [Run SIP](#SIP)
  - [LLaMA](#LLaMA) 
  - [MuSIc](#MuSIc) 
- [Run clarification need prediction](#Run-clarification-need-prediction) 
- [Evaluate SIP and clarification need prediction](#Evaluate-SIP-and-clarification-need-prediction) 
- [Run action prediction](#Run-action-prediction) 
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
## Run SIP
### LLaMA
We provide the script for running LLaMA; see [here](./model/LLaMA.py).
Before running the script, please first download the LLaMA original checkpoints and convert them to the Hugging Face Transformers format; see [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
Because LLaMA performs extremely badly on Chinese text, we use the Chinese versions of LLaMA from [here](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md). 
Please follow the link to produce Chinese LLaMA checkpoints.
Note that only LLaMA-7B and 13B are available for Chinese LLaMA at the time of writing.

#### WISE
Run the following commands to run LLaMA on WISE:
```bash
python -u ./model/LLaMA.py \
--model LLaMA-zh-7B-plus \
--pretained {your local path to the checkpoint of Chinese LLaMA-7B-plus} \
--demonstration_path ./dataset/WISE/train_WISE.pkl \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--max_new_tokens 5 \
--batch_size 2 \
--demonstration_num 2

python -u ./model/LLaMA.py \
--model LLaMA-zh-13B-plus
--pretained {your local path to the checkpoint of Chinese LLaMA-13B-plus} \
--demonstration_path ./dataset/WISE/train_WISE.pkl \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--max_new_tokens 5 \
--batch_size 2 \
--demonstration_num 2
```
The output files would be saved in the paths `.\output\WISE.SIP.LLaMA-zh-7B-plus` and `.\output\WISE.SIP.LLaMA-zh-13B-plus`.

#### MSDialog
Run the following commands to run LLaMA on MSDialog:
```bash
python -u ./model/LLaMA.py \
--model LLaMA-7B \
--pretained {your local path to the checkpoint of LLaMA-7B} \
--demonstration_path ./dataset/MSDialog/train_MSDialog.pkl \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--max_new_tokens 10 \
--batch_size 4 \
--demonstration_num 2

python -u ./model/LLaMA.py \
--model LLaMA-13B \
--pretained {your local path to the checkpoint of LLaMA-13B} \
--demonstration_path ./dataset/MSDialog/train_MSDialog.pkl \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--max_new_tokens 10 \
--batch_size 2 \
--demonstration_num 2

python -u ./model/LLaMA.py \
--model LLaMA-30B \
--pretained {your local path to the checkpoint of LLaMA-30B} \
--demonstration_path ./dataset/MSDialog/train_MSDialog.pkl \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--max_new_tokens 10 \
--batch_size 1 \
--demonstration_num 2

python -u ./model/LLaMA.py \
--model LLaMA-65B \
--pretained {your local path to the checkpoint of LLaMA-65B} \
--demonstration_path ./dataset/MSDialog/train_MSDialog.pkl \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--max_new_tokens  10 \
--batch_size 1 \
--demonstration_num 2
```
The output files would be saved in the paths `.\output\MSDialog.SIP.LLaMA-7B`, `MSDialog.SIP.LLaMA-13B`, `MSDialog.SIP.LLaMA-30B` and `MSDialog.SIP.LLaMA-65B`.

### MuSIc
#### WISE
Train MuSIc on the training set of WISE and conduct inference on the validation and test sets of WISE:
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
The above commands would produce model checkpoints and inference output files, which are stored in the path `./output/WISE.SIP.DistanceCRF/checkpoints` and `./output/WISE.SIP.DistanceCRF/`, respectively.

#### MSDialog
Train MuSIc on the training set of MSDialog and conduct inference on the validation and test sets of MSDialog:
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
The above commands would produce model checkpoints and inference output files, which are stored in the path `./output/MSDialog.SIP.DistanceCRF/checkpoints` and `./output/MSDialog.SIP.DistanceCRF/`, respectively.

## Run clarification need prediction
Run the following command to train MuSIc on the training set of ClariQ:
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/train_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train \
# --initialization_path {your local path to the checkpoint trained on SIP on MSDialog}
```
The above commands would produce model checkpoints in the path `./output/ClariQ.SIP.DistanceCRF/checkpoints/`.
If you would like to fine-tune the checkpoint that is pre-trained on the SIP task on ClariQ, please specify `--initialization_path`. 
In this case, the checkpoints would be saved in the path `./output/ClariQ.SIP.DistanceCRF-TransferLearning/checkpoints/`.

Run the following command to infer MuSIc without pertaining on MSDialog on the validation and test sets of ClariQ:
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/valid_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference \
# --initialization_path {your local path to the checkpoint trained on SIP on MSDialog}

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/test_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference \
# --initialization_path {your local path to the checkpoint trained on SIP on MSDialog}
```
The above commands would produce inference output files in the path `./output/ClariQ.SIP.DistanceCRF/`.
If you would like to infer MuSIc with pertaining on MSDialog on the validation and test sets of ClariQ, please still specify `--initialization_path` that is used during training.


## Evaluate SIP and clarification need prediction
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.LLaMA-zh-7B-plus \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.LLaMA-zh-13B-plus \
--label_path ./dataset/WISE/test_WISE.pkl
```

```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.yesno-LLaMA-7B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.yesno-LLaMA-13B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.yesno-LLaMA-30B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.yesno-LLaMA-65B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```

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

## Run action prediction

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