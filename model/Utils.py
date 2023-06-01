from torch.nn.init import *
import torch.nn as nn
import numpy as np
import random
import time
import os

def hamming_score(y_true, y_pred):
    return ((y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)).mean()

def replicability(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # Sets the seed for generating random numbers. Returns a torch.Generator object.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.insufficient to get determinism
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def rounder(num, places=2):
    num=num*100
    return round(num, places)


def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def universal_sentence_embedding(sentences, mask, sqrt=True):
    '''
    :param sentences: [batch_size, seq_len, hidden_size]
    :param mask: [batch_size, seq_len]
    :param sqrt:
    :return: [batch_size, hidden_size]
    '''
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)

    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()

    sentence_sums /= divisor
    return sentence_sums


class Config():
    def __init__(self, args):
        if args.dataset == "WISE":
            self.max_target_length = 4  # (3+1)
            self.pad_index = 0
            self.soa_index = 1
            self.eoa_index = 2

            self.action2id = {'[PAD]': 0, '[SOA]': 1, '[EOA]': 2, 'Clarify-Choice': 3, 'Clarify-Yes_no': 4,
                              'Clarify-Open': 5, 'Request-Feedback': 6,
                              'Request-Rephrase': 7, 'Revise': 8, 'Recommend': 9, 'Answer-Summary': 10,
                              'Answer-Link': 11,
                              'Answer-NoAnswer': 12, 'Answer-fact-list': 13, 'Answer-fact-link': 14,
                              'Answer-fact-text': 15,
                              'Answer-fact-steps': 16, 'Answer-open-list': 17, 'Answer-open-link': 18,
                              'Answer-open-text': 19, 'Answer-open-steps': 20, 'Answer-opinion-text': 21,
                              'Answer-opinion-list': 22, 'Answer-opinion-link': 23, 'Unknown': 24, 'Chitchat': 25}  # 26

            self.id2action = {0: '[PAD]', 1: '[SOA]', 2: '[EOA]', 3: 'Clarify-Choice', 4: 'Clarify-Yes_no',
                         5: 'Clarify-Open', 6: 'Request-Feedback',
                         7: 'Request-Rephrase', 8: 'Revise', 9: 'Recommend', 10: 'Answer-Summary', 11: 'Answer-Link',
                         12: 'Answer-NoAnswer', 13: 'Answer-fact-list', 14: 'Answer-fact-link',
                         15: 'Answer-fact-text',
                         16: 'Answer-fact-steps', 17: 'Answer-open-list', 18: 'Answer-open-link',
                         19: 'Answer-open-text', 20: 'Answer-open-steps', 21: 'Answer-opinion-text',
                         22: 'Answer-opinion-list', 23: 'Answer-opinion-link', 24: 'Unknown', 25: 'Chitchat'}  # 26

            self.action = ['Clarify-Choice', 'Clarify-Yes_no', 'Clarify-Open', 'Request-Feedback',
                                'Request-Rephrase', 'Revise', 'Recommend', 'Answer-Summary', 'Answer-Link',
                                'Answer-NoAnswer', 'Answer-fact-list', 'Answer-fact-link', 'Answer-fact-text',
                                'Answer-fact-steps', 'Answer-open-list', 'Answer-open-link',
                                'Answer-open-text', 'Answer-open-steps', 'Answer-opinion-text',
                                'Answer-opinion-list', 'Answer-opinion-link', 'Unknown', 'Chitchat']

            self.action_I =['Clarify-Choice', 'Clarify-Yes_no', 'Clarify-Open', 'Request-Feedback',
                                'Request-Rephrase', 'Revise', 'Recommend']

            self.action_N = ['Answer-Summary', 'Answer-Link',
                                'Answer-NoAnswer', 'Answer-fact-list', 'Answer-fact-link', 'Answer-fact-text',
                                'Answer-fact-steps', 'Answer-open-list', 'Answer-open-link',
                                'Answer-open-text', 'Answer-open-steps', 'Answer-opinion-text',
                                'Answer-opinion-list', 'Answer-opinion-link', 'Unknown', 'Chitchat']


        elif args.dataset == "MSDialog":
            self.pad_index =0
            self.soa_index = 1
            self.eoa_index = 2
            self.max_target_length = 8  # (7+1)
            self.action2id = {'[PAD]': 0, '[SOA]': 1, '[EOA]': 2, 'OQ': 3, 'RQ': 4, 'CQ': 5, 'FQ': 6, 'IR': 7, 'FD': 8,
                              'PA': 9,
                              'PF': 10, 'NF': 11, 'GG': 12, 'JK': 13, 'O': 14}  # 15
            self.id2action = {0: '[PAD]', 1: '[SOA]', 2: '[EOA]', 3: 'OQ', 4: 'RQ', 5: 'CQ', 6: 'FQ', 7: 'IR', 8: 'FD',
                         9: 'PA', 10: 'PF', 11: 'NF', 12: 'GG', 13: 'JK', 14: 'O'}  # 15

            self.action = ['OQ', 'RQ', 'CQ', 'FQ', 'IR', 'FD', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']

            self.action_I= ['OQ', 'RQ', 'CQ', 'FQ', 'IR']
            self.action_N = ['FD', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
        else:
            self.action =None
            self.max_target_length=1
            self.action2id = {'[PAD]': 0, '[SOA]': 1, '[EOA]': 2}