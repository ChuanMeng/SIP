import sys
sys.path.append('./')

import argparse
from transformers import get_constant_schedule
import os
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MultiLabelBinarizer

from model.Dataset import Dataset, collate_fn
from model.Model import BILSTMCRF, ActionPrediction
from model.Trainer import Trainer
from model.Utils import replicability, Config

def train(args):
    config = Config(args)
    mlb = MultiLabelBinarizer(classes=config.action)

    conversations = torch.load(args.input_path)

    if args.task=="SIP":
        model = BILSTMCRF(args)

        model_optimizer = optim.Adam(
            [{"params": model.utterance_encoding.parameters()},
            {"params": model.posterior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_e_project.parameters()},
            {"params": model.posterior_e_project.parameters()},
            {"params": model.crf.parameters(), "lr": args.lr_crf}], args.lr)

    elif args.task in ["AP", "SIP-AP"]:
        model = ActionPrediction(args,config,mlb)
        model_optimizer = optim.Adam(model.parameters(), args.lr)
    else:
        raise NotImplementedError

    if args.initialization_path is not None:
        model.load_state_dict(torch.load(args.initialization_path)["model"])

    model_scheduler = get_constant_schedule(model_optimizer)

    writer = SummaryWriter(args.log_path)

    trainer = Trainer(args, model, writer)
    model_optimizer.zero_grad()

    for i in range(1, args.epoch_num+1):
        dataset = Dataset(args, config, mlb, conversations)
        trainer.train_epoch(dataset, collate_fn, i, model_optimizer, model_scheduler)
        trainer.serialize(i, model_scheduler, saved_model_path=args.saved_model_path)
    writer.close()

def inference(args):
    config = Config(args)
    mlb = MultiLabelBinarizer(classes=config.action)

    conversations = torch.load(args.input_path)
    dataset = Dataset(args, config, mlb, conversations)

    for epoch_id in range(1,args.epoch_num+1):
        checkpoint_path= args.saved_model_path + "/"+ str(epoch_id) + '.pkl'

        if os.path.exists(checkpoint_path):
            if args.task == "SIP":
                model = BILSTMCRF(args)
            elif args.task in ["AP", "SIP-AP"]:
                model = ActionPrediction(args,config,mlb)
            else:
                raise NotImplementedError

            model.load_state_dict(torch.load(checkpoint_path)["model"])
            trainer = Trainer(args, model)
            print('Inference on the {} dataset'.format(args.dataset))
            trainer.infer(str(epoch_id), dataset, collate_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default='SIP') # [SIP, AP, SIP-AP]
    parser.add_argument("--model", type=str) # [DistanceCRF, mlc, sg]
    parser.add_argument("--mode", type=str, default='train') # [train, inference]

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--saved_model_path", type=str)
    parser.add_argument("--initialization_path", type=str, default=None)
    parser.add_argument("--SIP_path", type=str) # only for action prediction
    parser.add_argument("--Oracle_SIP", action='store_true')

    parser.add_argument("--epoch_num", type=int, default=20)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_crf", type=float, default= 1e-3)
    parser.add_argument("--lr", type=float, default= 2e-5)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--inference_batch_size", type=int, default=1)

    parser.add_argument("--max_utterance_len", type=int, default=128)
    parser.add_argument("--max_context_len", type=int, default=384)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--BiLSTM_layers", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    replicability(seed=args.random_seed)

    if "WISE" in args.input_path:
        args.dataset= "WISE"
    elif "MSDialog" in args.input_path:
        args.dataset = "MSDialog"
    elif "ClariQ" in args.input_path:
        args.dataset = "ClariQ"
    else:
        raise NotImplementedError

    args.name = f"{args.dataset}.{args.task}.{args.model}"

    if args.initialization_path is not None:
        args.name = f"{args.dataset}.{args.task}.{args.model}-TL"
    else:
        args.name = f"{args.dataset}.{args.task}.{args.model}"

    args.output_path = os.path.join(args.output_path, args.name)
    args.saved_model_path = os.path.join(args.output_path, "checkpoints")
    args.log_path = os.path.join(args.log_path, args.name)

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'train':
        train(args)
    else:
        raise NotImplementedError