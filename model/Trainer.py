from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import json
import os
from collections import defaultdict
import torch
import codecs
import time
import sys
from model.Utils import rounder
import numpy as np

class Trainer(object):
    def __init__(self, args, model, writer=None):
        super(Trainer, self).__init__()
        self.args = args

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model

        self.accumulation_count = 0
        self.writer = writer

    def train_batch(self, epoch, data, optimizer, scheduler=None):
        self.accumulation_count += 1
        loss = self.model(data)

        sum_loss = torch.cat([loss[name].reshape(1) for name in loss]).sum()/self.args.accumulation_steps
        sum_loss.backward()

        if self.accumulation_count % self.args.accumulation_steps == 0:
            if self.args.task=="SIP":
                self.writer.add_scalar('Loss/overall', sum_loss.item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalar('Loss/crf', loss["loss_crf"].item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalar('Loss/mle_e', loss["loss_mle_e"].item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalars('Loss/all', {'overall': sum_loss.item(),'crf': loss["loss_crf"].item(),'mle_e': loss["loss_mle_e"].item()},scheduler.state_dict()['_step_count'])
            elif self.args.task in ["AP", "SIP-AP"]:
                self.writer.add_scalar('Loss', sum_loss.item(), scheduler.state_dict()['_step_count'])
            else:
                raise NotImplementedError

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        loss_dict = dict()
        for i in loss:
            loss_dict[i] = loss[i].cpu().item()

        return loss_dict

    def serialize(self, epoch, scheduler, saved_model_path):
        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}
        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        print("Saved epoch {} model".format(epoch))

    def train_epoch(self, train_dataset, train_collate_fn, epoch, optimizer, scheduler=None):
        self.model.train()  

        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.args.train_batch_size, shuffle=True)

        start_time = time.perf_counter()
        step = 0

        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            loss_dict = self.train_batch(epoch, data, optimizer=optimizer, scheduler=scheduler)
            step += 1

            if j >= 0 and j % 100 == 0:
                elapsed_time = time.perf_counter() - start_time
                print('Training: {} on the {} dataset'.format(self.args.name, self.args.dataset))
                print('Epoch:{}, Step:{}, Loss:{}, Time:{}, LR:{}'.format(epoch, scheduler.state_dict()['_step_count'],loss_dict, rounder(elapsed_time, 2), scheduler.get_last_lr()))
                sys.stdout.flush()

        sys.stdout.flush()

    def infer(self, epoch_id, dataset, collate_fn):
        self.eval_model.eval()
        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.inference_batch_size,shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_example_id = []
            accumulative_prediction = []

            for k, data in enumerate(test_loader, 0):
                if (k + 1) == 1 or (k + 1) % 100 == 0:
                    print("{} on the {} dataset: doing {} / total {} in epoch {}".format(self.args.name,self.args.dataset, k + 1,len(test_loader), epoch_id))

                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                # [pair_num, ?]
                predicted = self.eval_model(data)

                assert len(predicted)==len(data["example_id"])

                for idx, example_id in enumerate(data["example_id"]):
                    accumulative_example_id.append(example_id)
                    if self.args.task=="SIP":
                        accumulative_prediction.append("Initiative" if int(predicted[idx][-1])==1 else "Non-initiative")
                    elif self.args.task in ["AP", "SIP-AP"]:
                        accumulative_prediction.append(predicted[idx])
                    else:
                        raise NotImplementedError

            with open(os.path.join(self.args.output_path, str(epoch_id)+".txt"), 'w') as w:
                for index, example_id in enumerate(accumulative_example_id):
                    if self.args.task == "SIP":
                        w.write(example_id + '\t' + str(accumulative_prediction[index])  + '\n')
                    elif self.args.task in ["AP", "SIP-AP"]:
                        assert isinstance(accumulative_prediction[index], list)
                        w.write(example_id + '\t' + ",".join(accumulative_prediction[index]) + '\n')
                    else:
                        raise NotImplementedError

        return None

        