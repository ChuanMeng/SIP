from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import random
from transformers import BertTokenizer


class Dataset(Dataset):
    def __init__(self, args, config, mlb, conversations):
        super(Dataset, self).__init__()

        self.args = args
        self.config = config
        self.conversations = conversations

        self.mlb = mlb

        if self.args.dataset == "WISE":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif self.args.dataset in ["MSDialog", "ClariQ"]:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        if self.args.mode=="inference" and args.task == "SIP-AP" and self.args.Oracle_SIP==False:
            self.id2SIP={}
            with open(self.args.SIP_path, 'r') as r:
                for line in r:
                    example_id, prediction = line.rstrip().split()
                    self.id2SIP[example_id]=prediction

        self.conversations_tensor = []
        self.load()

    def load(self):
        for conversation_index, conversation in enumerate(self.conversations):
            conversation_content = {"example_id": [], "user_utterance": [], "user_I_label": [], "system_utterance": [],
                                    "system_I_label": [], "context": [], "system_action_label": [],
                                    "system_action_sequence": [],"system_I_prediction":[]}

            context_list = []

            for example_index, example in enumerate(conversation):
                # process SIP
                conversation_content["example_id"].append(example["example_id"])
                conversation_content["user_utterance"].append(torch.tensor(self.tokenizer.encode(example["user_utterance"], add_special_tokens=True, max_length=self.args.max_utterance_len, truncation=True,padding='max_length')))
                conversation_content["user_I_label"].append(torch.tensor(1) if example["user_I_label"]=="Initiative" else torch.tensor(0))
                conversation_content["system_utterance"].append(torch.tensor(self.tokenizer.encode(example["system_utterance"], add_special_tokens=True, max_length=self.args.max_utterance_len, truncation=True,padding='max_length')))
                conversation_content["system_I_label"].append(torch.tensor(1) if example["system_I_label"]=="Initiative" else torch.tensor(0))

                if self.args.mode == "inference" and self.args.task == "SIP-AP" and self.args.Oracle_SIP==False:
                    conversation_content["system_I_prediction"].append(torch.tensor(1) if self.id2SIP[example["example_id"]]== "Initiative" else torch.tensor(0))
                else:
                    conversation_content["system_I_prediction"]=conversation_content["system_I_label"]

                # process context
                context_list.append(example["user_utterance"])
                assert len(context_list) == example_index * 2 + 1

                context_text = " ".join(context_list)

                context_tokens = self.tokenizer.tokenize(context_text)

                if len(context_tokens) > (self.args.max_context_len - 2):
                    context_tokens_ = context_tokens[-(self.args.max_context_len - 2):]  # 510 tokens
                    context_tokens_ = ['[CLS]'] + context_tokens_ + ['[SEP]']
                else:
                    context_tokens_ = ['[CLS]'] + context_tokens + ['[SEP]'] + ['[PAD]'] * (self.args.max_context_len - 2 - len(context_tokens))

                assert len(context_tokens_) == self.args.max_context_len

                context_id = self.tokenizer.convert_tokens_to_ids(context_tokens_)
                conversation_content["context"].append(torch.tensor(context_id))
                context_list.append(example["system_utterance"])

                # process system_action_label
                conversation_content["system_action_label"].append(torch.tensor(self.mlb.fit_transform([example["system_action"]])).squeeze())  # [label_size]

                # process system_action
                system_action_sequence = example["system_action"] + ['[EOA]']
                if len(system_action_sequence) < self.config.max_target_length:
                    system_action_sequence = system_action_sequence + ['[PAD]'] * (self.config.max_target_length - len(system_action_sequence))
                assert len(system_action_sequence) == self.config.max_target_length
                conversation_content["system_action_sequence"].append(torch.tensor([self.config.action2id[action] for action in system_action_sequence]))


            assert len(conversation_content["user_utterance"]) == len(conversation_content["user_I_label"]) == len(
                conversation_content["system_utterance"]) == len(conversation_content["system_I_label"])== len(
                conversation_content["context"]) == len(conversation_content["system_action_label"]) == len(
                conversation_content["system_action_sequence"])==len(conversation_content["system_I_prediction"])

            user_utterance_conversation = torch.stack(conversation_content["user_utterance"])  # [?, max_utterance_len]
            user_I_label_conversation = torch.stack(conversation_content["user_I_label"])  # [?]
            system_utterance_conversation = torch.stack(conversation_content["system_utterance"])  # [?, max_utterance_len]
            system_I_label_conversation = torch.stack(conversation_content["system_I_label"])  # [?]
            system_I_prediction_conversation = torch.stack(conversation_content["system_I_prediction"])  # [?]
            #
            context_conversation = torch.stack(conversation_content["context"])  # [?, max_context_len]
            system_action_label_conversation = torch.stack(conversation_content["system_action_label"])  # [?, label_size]
            system_action_sequence_conversation = torch.stack(conversation_content["system_action_sequence"])  # [?, max_target_length]

            self.conversations_tensor.append(
                [conversation_content["example_id"],
                 user_utterance_conversation,
                 user_I_label_conversation,
                 system_utterance_conversation,
                 system_I_label_conversation,
                 #
                 context_conversation,
                 system_action_label_conversation,
                 system_action_sequence_conversation,
                 #
                 system_I_prediction_conversation
                 ])

            self.len = conversation_index + 1

    def __getitem__(self, index):
        conversation_tensor = self.conversations_tensor[index]
        return [conversation_tensor[0], conversation_tensor[1], conversation_tensor[2], conversation_tensor[3], conversation_tensor[4], conversation_tensor[5], conversation_tensor[6], conversation_tensor[7], conversation_tensor[8]]

    def __len__(self):
        return self.len


def collate_fn(data):
    example_id, user_utterance_conversations, user_I_label_conversations, system_utterance_conversations, system_I_label_conversations, context_conversations, system_action_label_conversations, system_action_sequence_conversations, system_I_prediction_conversations = zip(*data)
    return {'example_id': example_id[-1], # [batch]
            'user_utterance': torch.stack(user_utterance_conversations),  # [batch, ?, max_utterance_len]
            'user_I_label': torch.stack(user_I_label_conversations),  # [batch, ?]
            'system_utterance': torch.stack(system_utterance_conversations),  # [batch, ?, max_utterance_len]
            'system_I_label': torch.stack(system_I_label_conversations),  # [batch, ?]
            #
            'context': torch.stack(context_conversations),  # [batch, ?, max_context_len],
            'system_action_label': torch.stack(system_action_label_conversations),  # [batch, ?, label_size]
            'system_action_sequence': torch.stack(system_action_sequence_conversations),
            #
            'system_I_prediction': torch.stack(system_I_prediction_conversations),
            }

