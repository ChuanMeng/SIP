import sys
import os
import codecs
from sys import *
import random
import pprint
import pandas as pd
from tqdm import tqdm
import json
import argparse
import torch

def preprocess_MSDialog(file_path):
    MSDialog = {}
    conversation_index = 0
    utterances = []
    utterance_pos = 0

    with open(file_path) as r:
        for line in r:
            if line != '\n':
                columns = line.strip().split('\t')

                utterance_pos += 1
                tags = columns[0].split('_')
                if len(tags) > 1 and 'GG' in tags:
                    tags.remove('GG')
                if len(tags) > 1 and 'O' in tags:
                    tags.remove('O')
                if len(tags) > 1 and 'JK' in tags:
                    tags.remove('JK')

                utterance = {}
                utterance["tags"] = (" ").join(tags)
                utterance["utterance"] = columns[1]
                utterance["actor_type"] = columns[2]
                utterance["utterance_pos"] = utterance_pos
                utterances.append(utterance)

            if line == '\n':
                MSDialog[str(conversation_index)] = {"utterances": utterances}
                conversation_index += 1
                utterances = []
                utterance_pos = 0


    conversation_id = 0
    conversations = []

    for index, conv in MSDialog.items():
        conversation = []
        conversation_id += 1

        current_role = "N/A"
        merged_conv = []
        merged_role_turn = []

        # merge concercutive utterances
        for i, turn in enumerate(conv['utterances']):
            if current_role != turn["actor_type"]:
                if i == 0:
                    merged_role_turn.append(turn)
                else:
                    merged_conv.append(merged_role_turn)
                    merged_role_turn = []
                    merged_role_turn.append(turn)

                current_role = turn["actor_type"]
            else:
                merged_role_turn.append(turn)

            if i == len(conv['utterances']) - 1:
                merged_conv.append(merged_role_turn)

        # traverse each turn
        system_action_count_training = {'PA': 2390, 'IR': 646, 'FD': 626, 'RQ': 272, 'CQ': 257, 'FQ': 209, 'GG': 179,
                                        'PF': 154, 'NF': 128, 'OQ': 72, 'JK': 60, 'O': 11}

        for i, merged_turn in enumerate(merged_conv):

            example = dict()
            merged_utterance = ""

            if merged_turn[0]['actor_type'] == "User":
                merged_tags = []

                for j, turn in enumerate(merged_turn):
                    merged_tags = merged_tags + turn["tags"].split()
                    merged_utterance = merged_utterance + turn['utterance'] if merged_utterance == "" else merged_utterance + " " + turn['utterance']

                if 'OQ' in merged_tags or 'RQ' in merged_tags or 'CQ' in merged_tags or 'FQ' in merged_tags or 'IR' in merged_tags:
                    #I_label = "I"
                    I_label = "Initiative"
                else:
                    #I_label = "N"
                    I_label = "Non-initiative"

                example["conversation_id"] = conversation_id
                example["utterance_id"] = str(i + 1)
                example["role"] = "user"
                example["utterance"] = merged_utterance
                example["actions"] = list(set(merged_tags))
                example["I_label"] = I_label

            elif merged_turn[0]['actor_type'] == "Agent":
                merged_tags = []

                for j, turn in enumerate(merged_turn):
                    merged_tags = merged_tags + turn["tags"].split()
                    merged_utterance = merged_utterance + turn[
                        'utterance'] if merged_utterance == "" else merged_utterance + " " + turn['utterance']

                if 'OQ' in merged_tags or 'RQ' in merged_tags or 'CQ' in merged_tags or 'FQ' in merged_tags or 'IR' in merged_tags:
                    #I_label = "I"
                    I_label = "Initiative"
                else:
                    #I_label = "N"
                    I_label = "Non-initiative"

                example["conversation_id"] = conversation_id
                example["utterance_id"] = str(i + 1)
                example["role"] = "system"
                example["utterance"] = merged_utterance
                example["actions"] = list(set(merged_tags))
                example["actions"].sort(reverse=True, key=lambda k: system_action_count_training[k])
                example["I_label"] = I_label
            else:
                raise error

            conversation.append(example)

        conversations.append(conversation)

    print('total conversations:', len(conversations))
    print('total_number_merged_turns:', sum([len(conversation) for conversation in conversations]))
    print('the lowest length of conversations:', min([len(conversation) for conversation in conversations]))
    print('the maximum length of conversations:', max([len(conversation) for conversation in conversations]))
    print('average length of a conversation:',(sum([len(conversation) for conversation in conversations])) / len(conversations))

    # statistic about action
    actions = []
    actions_system = []
    actions_user = []
    for conversation in conversations:
        for turn in conversation:
            if turn["role"] == "user":
                actions.append(len(turn["actions"]))
                actions_user.append(len(turn["actions"]))
            elif turn["role"] == "system":
                actions.append(len(turn["actions"]))
                actions_system.append(len(turn["actions"]))
            else:
                raise exception

    assert sum(actions_system) + sum(actions_user) == sum(actions)
    assert len(actions_system) + len(actions_user) == len(actions)  # the total turns

    print('The minimum number of actions per turn:', min(actions))
    print('The maximum number of actions per turn:', max(actions))
    print('The average number of actions per turn:', sum(actions) / len(actions))

    print('The minimum number of actions per user turn:', min(actions_user))
    print('The maximum number of actions per user turn:', max(actions_user))
    print('The average number of actions per user turn:', sum(actions_user) / len(actions_user))

    print('The minimum number of actions per system turn:', min(actions_system))
    print('The maximum number of actions per system turn:', max(actions_system))
    print('The average number of actions per system turn:', sum(actions_system) / len(actions_system))

    # second stage prepocess
    conversations_ = []
    for conversation in conversations:
        conversation_ = []
        if len(conversation) == 1:
            print(conversation)
            continue
        for id, example in enumerate(conversation):
            # if there is a system's utterance at the first position of a conversation, it will be dropped.
            # if there is a following user merged utterance after the system utterance, it will be dropped.
            if example["role"] == "system" and id >= 1:
                example_ = dict()

                example_["conversation_id"] = example["conversation_id"]
                example_["example_id"] = str(example["conversation_id"]) + "(" + conversation[id - 1][
                    "utterance_id"] + "_" + example["utterance_id"] + ")"

                assert conversation[id - 1]["role"] == "user"
                example_["user_utterance"] = conversation[id - 1]["utterance"]
                example_["user_action"] = conversation[id - 1]["actions"]
                example_["user_I_label"] = conversation[id - 1]["I_label"]

                example_["system_utterance"] = example["utterance"]
                example_["system_action"] = example["actions"]
                example_["system_I_label"] = example["I_label"]

                conversation_.append(example_)
        conversations_.append(conversation_)

    print('total conversations_:', len(conversations_))
    print('total_number of examples (pairs):', sum([len(conversation_) for conversation_ in conversations_]))
    print('the minimum number of pairs in a conversation:',min([len(conversation_) for conversation_ in conversations_]))
    print('the maximum number of pairs in a conversation:',max([len(conversation_) for conversation_ in conversations_]))
    print('the average number of pairs in a conversation:',(sum([len(conversation_) for conversation_ in conversations_])) / len(conversations_))
    return conversations_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    conversations = preprocess_MSDialog(args.input_path)
    torch.save(conversations, args.output_path)

