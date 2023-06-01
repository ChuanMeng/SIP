import sys
import os
import codecs
from sys import *
import random
import pprint
from tqdm import tqdm
import json
import torch
import argparse

def preprocess_WISE(file_path):
    conversation_id = 0
    conversations = []

    with codecs.open(file_path, encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)

            conversation = []
            conversation_id += 1

            current_role = "N/A"
            merged_conv = []
            merged_role_turn = []

            # merge concercutive utterances
            for i, turn in enumerate(conv['conversations']):
                # unify action and intent
                if turn["role"] == "user":
                    turn["unified_intent"] = ['-'.join(turn['intent'])]
                elif turn["role"] == "system":
                    turn["unified_action"] = ['-'.join(turn['action'])]
                else:
                    raise exception

                if current_role != turn["role"]:
                    if i == 0:
                        merged_role_turn.append(turn)
                    else:
                        merged_conv.append(merged_role_turn)
                        merged_role_turn = []
                        merged_role_turn.append(turn)

                    current_role = turn["role"]
                else:
                    merged_role_turn.append(turn)

                if i == len(conv['conversations']) - 1:
                    merged_conv.append(merged_role_turn)

            # traverse each turn
            system_action_count_training = {'Chitchat': 1343, 'Answer-open-text': 1239, 'Answer-open-link': 920,
                                            'Answer-open-list': 764, 'Answer-NoAnswer': 332, 'Answer-fact-text': 326,
                                            'Clarify-Choice': 317, 'Clarify-Yes_no': 181, 'Answer-opinion-text': 153,
                                            'Clarify-Open': 113, 'Answer-fact-list': 87, 'Answer-opinion-list': 64,
                                            'Answer-fact-link': 61, 'Request-Rephrase': 49, 'Answer-open-steps': 33,
                                            'Unknown': 22, 'Recommend': 21, 'Answer-opinion-link': 15,
                                            'Answer-fact-steps': 10, 'Request-Feedback': 7, 'Revise': 4}

            for i, merged_turn in enumerate(merged_conv):
                example = dict()
                merged_utterance = ""

                if merged_turn[0]["role"] == "user":
                    merged_intent = []
                    merged_unified_intent = []

                    for j, turn in enumerate(merged_turn):
                        merged_intent = merged_intent + turn["intent"]
                        merged_unified_intent = merged_unified_intent + turn["unified_intent"]
                        merged_utterance = merged_utterance + turn['response'] if merged_utterance == "" else merged_utterance + " " + turn['response']

                    if "Request" in merged_intent or "Revise" in merged_intent or "Reveal" in merged_intent:
                        #I_label = "I"
                        I_label = "Initiative"
                    else:
                        #I_label = "N"
                        I_label = "Non-initiative"

                    example["conversation_id"] = conversation_id
                    example["utterance_id"] = str(i + 1)
                    example["role"] = "user"
                    example["utterance"] = merged_utterance
                    example["actions"] = list(set(merged_unified_intent))
                    example["I_label"] = I_label

                else:
                    merged_action = []
                    merged_unified_action = []

                    for j, turn in enumerate(merged_turn):
                        merged_action = merged_action + turn['action']
                        merged_unified_action = merged_unified_action + turn['unified_action']
                        merged_utterance = merged_utterance + turn[
                            'response'] if merged_utterance == "" else merged_utterance + " " + turn['response']

                    if "Clarify" in merged_action or "Recommend" in merged_action or "Request" in merged_action or "Revise" in merged_action:
                        #I_label = "I"
                        I_label = "Initiative"
                    else:
                        #I_label = "N"
                        I_label = "Non-initiative"

                    example["conversation_id"] = conversation_id
                    example["utterance_id"] = str(i + 1)
                    example["role"] = "system"
                    example["utterance"] = merged_utterance
                    example["actions"] = list(set(merged_unified_action))
                    example["actions"].sort(reverse=True, key=lambda k: system_action_count_training[k])
                    example["I_label"] = I_label

                conversation.append(example)

            conversations.append(conversation)

    print('total conversations:', len(conversations))
    print('total_number_merged_turns:', sum([len(conversation) for conversation in conversations]))
    print('the lowest length of conversations:', min([len(conversation) for conversation in conversations]))
    print('the maximum length of conversations:', max([len(conversation) for conversation in conversations]))
    print('average length of a conversation:',
          (sum([len(conversation) for conversation in conversations])) / len(conversations))

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
    # id2utterance={}
    for conversation in conversations:
        conversation_ = []

        for id, example in enumerate(conversation):
            # if there is a following user merged utterance after the system utterance, it will be dropped.
            if example["role"] == "system":
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
    print('the minimum number of pairs in a conversation:', min([len(conversation_) for conversation_ in conversations_]))
    print('the maximum number of pairs in a conversation:', max([len(conversation_) for conversation_ in conversations_]))
    print('the average number of pairs in a conversation:',(sum([len(conversation_) for conversation_ in conversations_])) / len(conversations_))

    return conversations_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    conversations = preprocess_WISE(args.input_path)
    torch.save(conversations, args.output_path)