import sys
import os
import codecs
from sys import *
import random
import pprint
from tqdm import tqdm
import json
import argparse
import torch

def preprocess_ClariQ(file_path):
    ClariQ={}
    with open(file_path) as r:
        for line in r:
            columns = line.strip().split('\t')
            if columns[0]=="topic_id":
                continue
            elif columns[0] in ClariQ:
                # only pick one row per topic
                if columns[6]!="Q00001":
                    ClariQ[columns[0]]["question"] = columns[7]
                continue

            else:
                assert columns[6]!="Q00001"
                ClariQ[columns[0]]={"initial_request":columns[1], "clarification_need":columns[3], "question": columns[7]}

    conversations = []
    I_num=0
    N_num=0

    for index, conv in ClariQ.items():
        conversation=[]

        example = dict()
        example["conversation_id"] = index
        example["utterance_id"] = str(1)
        example["role"] = "user"
        example["utterance"] = conv["initial_request"]
        example["actions"] = []
        example["I_label"] = "Initiative"

        conversation.append(example)

        example = dict()
        example["conversation_id"] = index
        example["utterance_id"] = str(2)
        example["role"] = "system"
        example["utterance"] = conv["question"] if int(conv["clarification_need"]) in [2,3,4] else "answer"
        example["actions"] = []
        example["I_label"] = "Initiative" if int(conv["clarification_need"]) in [2,3,4] else "Non-initiative"

        if int(conv["clarification_need"]) in [2, 3, 4]:
            I_num+=1
        elif int(conv["clarification_need"]) in [1]:
            N_num+=1
        else:
            raise Excepetion

        conversation.append(example)

        conversations.append(conversation)

    print('total conversations:', len(conversations))
    print('total_number_turns:', sum([len(conversation) for conversation in conversations]))
    print('the lowest length of conversations:', min([len(conversation) for conversation in conversations]))
    print('the maximum length of conversations:', max([len(conversation) for conversation in conversations]))
    print('average length of a conversation:', (sum([len(conversation) for conversation in conversations]))/len(conversations))

    # second stage prepocess
    conversations_=[]
    for conversation in conversations:
        conversation_=[]
        if len(conversation)==1:
            print(conversation)
            continue
        for id, example in enumerate(conversation):
            # if there is a system's utterance at the first position of a conversation, it will be dropped.
            # if there is a following user merged utterance after the system utterance, it will be dropped.
            if example["role"]=="system" and id>=1:
                example_ = dict()

                example_["conversation_id"]=example["conversation_id"]
                example_["example_id"]=str(example["conversation_id"])+"("+conversation[id - 1]["utterance_id"]+"_"+ example["utterance_id"]+")"

                assert conversation[id-1]["role"]=="user"
                example_["user_utterance"]=conversation[id-1]["utterance"]
                example_["user_action"] = conversation[id-1]["actions"]
                example_["user_I_label"] = conversation[id-1]["I_label"]

                example_["system_utterance"]= example["utterance"]
                example_["system_action"] = example["actions"]
                example_["system_I_label"] = example["I_label"]

                conversation_.append(example_)

        conversations_.append(conversation_)

    print('total conversations_:', len(conversations_))
    print('total_number of examples (pairs):', sum([len(conversation_) for conversation_ in conversations_]))
    print('the minimum number of pairs in a conversation:', min([len(conversation_) for conversation_ in conversations_]))
    print('the maximum number of pairs in a conversation:', max([len(conversation_) for conversation_ in conversations_]))
    print('the average number of pairs in a conversation:', (sum([len(conversation_) for conversation_ in conversations_]))/len(conversations_))
    print(N_num, I_num)

    return conversations_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    conversations = preprocess_ClariQ(args.input_path)
    torch.save(conversations, args.output_path)


