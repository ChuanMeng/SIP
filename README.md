# System Initiative Prediction (SIP)
![](https://api.visitorbadge.io/api/VisitorHit?user=ChuanMeng&repo=QPP4CS&countColor=%237B1E7A)

This is the code repository for the paper titled **System Initiative Prediction for Multi-turn Conversational Information Seeking**.

We kindly ask you to cite our papers if you find this repository useful: 
```
@inproceedings{meng2023system,
 author = {Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
 title = {System Initiative Prediction for Multi-turn Conversational Information Seeking},
 booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
 year = {2023},
}
```

To reproduce the reported results in the paper, please follow the instructions outlined below:
- [Prerequisites](#Prerequisites)
- [Data Preprocessing](#Data-Preprocessing)
- [Run SIP](#Run-SIP)
  - [LLaMA](#LLaMA) 
  - [MuSIc](#MuSIc) 
- [Run clarification need prediction](#Run-clarification-need-prediction) 
- [Evaluate SIP and clarification need prediction](#Evaluate-SIP-and-clarification-need-prediction) 
- [Run action prediction](#Run-action-prediction) 
  - [Multi-label classification](#Multi-label-classification) 
  - [SIP+Multi-label classification](#Multi-label-classification) 
  - [Sequence generation](#Sequence-generation)
  - [SIP+Sequence generation](#Sequence-generation)
- [Evaluate action prediction](#Evaluate-Action-prediction)

Please be aware that, as the GitHub repository is anonymous, the hyperlinks in the outline are not functional. 
This issue will be rectified in the regular GitHub repository.

## Prerequisites
Install dependencies:  
```bash
pip install -r requirements.txt
```
## Data Preprocessing
The following commands are used to conduct data preprocessing, which includes the automatic annotation of initiative-taking decision labels. 
We derive the initiative annotations by mapping the manual annotations of actions to initiative or non-initiative labels.
The raw data for the WISE, MSDialog and ClariQ is stored in the paths `./data/WISE/`, `./data/MSDialog/` and `./data/ClariQ/`.
The preprocessed data would be still stored in these paths.

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

Before running the script, please make sure your Cuda version is greater than or equal to 11.1. 
Next, download the LLaMA original checkpoints and convert them to the Hugging Face Transformers format; see [here](https://huggingface.co/docs/transformers/main/model_doc/llama) for more details.

Because the original LLaMA performs extremely badly on WISE, which is in Chinese text, we use the Chinese versions of LLaMA from [here](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md) on WISE. 
Please follow the instruction in the link to produce Chinese LLaMA checkpoints.
At the time of writing, the Chinese versions of LLaMA only have **Chinese-LLaMA-7B**, **Chinese-LLaMA-13B**, **Chinese-LLaMA-Plus-7B** and **Chinese-LLaMA-Plus-13B**.
In particular, we choose to use the plus versions, namely **Chinese-LLaMA-Plus-7B** and **Chinese-LLaMA-Plus-13B**, because the plus versions were trained on more data and are highly recommended for use by the releaser.

Our preliminary experiments showed that all LLaMA variants on the two datasets perform best when injected with **2 complete conversations** randomly sampled from the training set, given the same random seed.

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
The inference output files would be saved in the paths `.\output\WISE.SIP.LLaMA-zh-7B-plus` and `.\output\WISE.SIP.LLaMA-zh-13B-plus`.

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
The inference output files would be saved in the paths `.\output\MSDialog.SIP.LLaMA-7B`, `MSDialog.SIP.LLaMA-13B`, `MSDialog.SIP.LLaMA-30B` and `MSDialog.SIP.LLaMA-65B`.

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
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/WISE.SIP.DistanceCRF/checkpoints/` and `./output/WISE.SIP.DistanceCRF/`, respectively.

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
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/MSDialog.SIP.DistanceCRF/checkpoints/` and `./output/MSDialog.SIP.DistanceCRF/`, respectively.

## Run clarification need prediction
Run the following command to directly train MuSIc on the training set of ClariQ and conduct inference on the validation and test sets of ClariQ:
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/train_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode train \

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/valid_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference \

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/test_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference \
```
The above commands would produce model checkpoints and inference output files, which are saved in the paths `./output/ClariQ.SIP.DistanceCRF/checkpoints/` and `./output/ClariQ.SIP.DistanceCRF/`, respectively.

Run the following command to fine-tune MuSIc (pre-trained on SIP on the training set of MSDialog) on the training set of ClariQ and conduct inference on the validation and test sets of ClariQ:
```bash
python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/train_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--initialization_path {your local path to the checkpoint trained on SIP on MSDialog} \
--mode train \

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/valid_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--initialization_path {your local path to the checkpoint trained on SIP on MSDialog} \
--mode inference \

python -u ./model/Run.py \
--task SIP \
--model DistanceCRF \
--input_path ./dataset/ClariQ/test_ClariQ.pkl \
--output_path ./output/ \
--log_path ./log/ \
--initialization_path {your local path to the checkpoint trained on SIP on MSDialog} \
--mode inference \
```
Please specify `--initialization_path`, which shows your local path to the checkpoint trained on SIP on MSDialog.
The above commands would produce checkpoints, which would be saved in the paths `./output/ClariQ.SIP.DistanceCRF-TransferLearning/checkpoints/`; the inference output files would be saved in the path `./output/ClariQ.SIP.DistanceCRF-TransferLearning/`.

## Evaluate SIP and clarification need prediction
Evaluate LLaMA on the test set of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.LLaMA-zh-7B-plus \
--label_path ./dataset/WISE/test_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.LLaMA-zh-13B-plus \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the paths `./output/WISE.SIP.LLaMA-zh-7B-plus/` and `./output/WISE.SIP.LLaMA-zh-13B-plus/`.

Evaluate LLaMA on the test set of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.LLaMA-7B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.LLaMA-13B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.LLaMA-30B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.LLaMA-65B \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the paths `./output/MSDialog.SIP.LLaMA-7B/`, `./output/MSDialog.SIP.LLaMA-13B/`, `./output/MSDialog.SIP.LLaMA-30B/` and `./output/MSDialog.SIP.LLaMA-65B/`.

Evaluate MuSIc on the validation and test sets of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.DistanceCRF \
--label_path ./dataset/WISE/valid_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP.DistanceCRF \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the path `./output/WISE.SIP.DistanceCRF/`.

Evaluate MuSIc on the validation and test sets of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.DistanceCRF \
--label_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP.DistanceCRF \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the path `./output/MSDialog.SIP.DistanceCRF/`.

Evaluate MuSIc (without pre-training on SIP) on the validation and test sets of ClariQ:
```bash
python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF \
--label_path ./dataset/ClariQ/valid_ClariQ.pkl

python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF \
--label_path ./dataset/ClariQ/test_ClariQ.pkl
```
The files recording the evaluation results would be saved in the path `./output/ClariQ.SIP.DistanceCRF/`.

Evaluate MuSIc (with pre-training on SIP) on the validation and test sets of ClariQ:
```bash
python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF-TransferLearning \
--label_path ./dataset/ClariQ/valid_ClariQ.pkl

python -u Evaluation.py \
--prediction_path ./output/ClariQ.SIP.DistanceCRF-TransferLearning \
--label_path ./dataset/ClariQ/test_ClariQ.pkl
```
The files recording the evaluation results would be saved in the path `./output/ClariQ.SIP.DistanceCRF-TransferLearning/`.

## Run action prediction

### Multi-label classification

#### WISE
Train the action prediction model based on multi-label classification on the training set of WISE and conduct inference on the validation and test sets of WISE:
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
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/WISE.AP.mlc/checkpoints/` and `./output/WISE.AP.mlc/`, respectively.

#### MSDialog
Train the action prediction model based on multi-label classification on the training set of MSDialog and conduct inference on the validation and test sets of MSDialog:
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
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/MSDialog.AP.mlc/checkpoints/` and `./output/MSDialog.AP.mlc/`, respectively.

### SIP+multi-label classification

#### WISE
Train the SIP-aware action prediction model based on multi-label classification on the training set of WISE and conduct inference on the validation and test sets of WISE:
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
--SIP_path {Your local path to MuSIc's best inference output file on SIP} \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path {Your local path to MuSIc's best inference output file on SIP} \
--mode inference
```
Note that before conducting inference, please specify `--SIP_path`, which shows the path to MuSIc's best inference output file on SIP.
E.g., for conducting inference on the test set of WISE, please specify MuSIc's best inference output file on the test set of WISE on the SIP task.
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/WISE.SIP-AP.mlc/checkpoints/` and `./output/WISE.SIP-AP.mlc/`, respectively.

#### MSDialog
Train the SIP-aware action prediction model based on multi-label classification on the training set of MSDialog and conduct inference on the validation and test sets of MSDialog:
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
--SIP_path {Your local path to MuSIc's best inference output file on SIP} \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model mlc \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path {Your local path to MuSIc's best inference output file on SIP} \
--mode inference
```
Note that before conducting inference, please specify `--SIP_path`, which shows the path to MuSIc's best inference output file on SIP.
E.g., for conducting inference on the test set of MSDialog, please specify MuSIc's best inference output file on the test set of MSDialog on the SIP task.
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/MSDialog.SIP-AP.mlc/checkpoints/` and `./output/MSDialog.SIP-AP.mlc/`, respectively.

### Sequence generation
#### WISE
Train the action prediction model based on sequence generation on the training set of WISE and conduct inference on the validation and test sets of WISE:
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
The above commands would produce model checkpoints and inference output files stored in the paths `./output/WISE.AP.sg/checkpoints/` and `./output/WISE.AP.sg/`, respectively.

#### MSDialog
Train the action prediction model based on sequence generation on the training set of MSDialog and conduct inference on the validation and test sets of MSDialog:
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
--mode inference

python -u ./model/Run.py \
--task AP \
--model sg \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--mode inference
```
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/MSDialog.AP.sg/checkpoints/` and `./output/MSDialog.AP.sg/`, respectively.

### SIP+sequence generation
#### WISE
Train the SIP-aware action prediction model based on sequence generation on the training set of WISE and conduct inference on the validation and test sets of WISE:
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
--SIP_path {Your local path to MuSIc's inference output file on SIP} \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/WISE/test_WISE.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path {Your local path to MuSIc's inference output file on SIP} \
--mode inference
```
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/WISE.SIP-AP.sg/checkpoints/` and `./output/WISE.SIP-AP.sg/`, respectively.

#### MSDialog
Train the SIP-aware action prediction model based on sequence generation on the training set of MSDialog and conduct inference on the validation and test sets of MSDialog:
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
--SIP_path {Your local path to MuSIc's inference output file on SIP} \
--mode inference

python -u ./model/Run.py \
--task SIP-AP \
--model sg \
--input_path ./dataset/MSDialog/test_MSDialog.pkl \
--output_path ./output/ \
--log_path ./log/ \
--SIP_path {Your local path to MuSIc's inference output file on SIP} \
--mode inference
```
The above commands would produce model checkpoints and inference output files, which are stored in the paths `./output/MSDialog.SIP-AP.sg/checkpoints/` and `./output/MSDialog.SIP-AP.sg/`, respectively.

## Evaluate action prediction
Evaluate the action prediction model based on multi-label classification on the validation and test sets of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.AP.mlc \
--label_path ./dataset/WISE/valid_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.AP.mlc \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the path `./output/WISE.AP.mlc/`.

Evaluate the action prediction model based on multi-label classification on the validation and test sets of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.mlc \
--label_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.mlc \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the path `./output/MSDialog.AP.mlc/`.

Evaluate the SIP-aware action prediction model based on multi-label classification on the validation and test sets of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.mlc \
--label_path ./dataset/WISE/valid_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.mlc \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the path `./output/WISE.SIP-AP.mlc/`.

Evaluate the SIP-aware action prediction model based on multi-label classification on the validation and test sets of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.mlc \
--label_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.mlc \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the path `./output/MSDialog.SIP-AP.mlc/`.

Evaluate the action prediction model based on sequence generation on the validation and test sets of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.AP.sg \
--label_path ./dataset/WISE/valid_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.AP.sg \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the path `./output/WISE.AP.sg/`.

Evaluate the action prediction model based on sequence generation on the validation and test sets of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.sg \
--label_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.AP.sg \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the path `./output/MSDialog.AP.sg/`.

Evaluate the SIP-aware action prediction model based on sequence generation on the validation and test sets of WISE:
```bash
python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.sg \
--label_path ./dataset/WISE/valid_WISE.pkl

python -u Evaluation.py \
--prediction_path ./output/WISE.SIP-AP.sg \
--label_path ./dataset/WISE/test_WISE.pkl
```
The files recording the evaluation results would be saved in the path `./output/WISE.SIP-AP.sg/`.

Evaluate the SIP-aware action prediction model based on sequence generation on the validation and test sets of MSDialog:
```bash
python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.sg \
--label_path ./dataset/MSDialog/valid_MSDialog.pkl

python -u Evaluation.py \
--prediction_path ./output/MSDialog.SIP-AP.sg \
--label_path ./dataset/MSDialog/test_MSDialog.pkl
```
The files recording the evaluation results would be saved in the path `./output/MSDialog.SIP-AP.sg/`.