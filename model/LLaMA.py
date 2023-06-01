import sys
sys.path.append('./')

from transformers import AutoTokenizer, LlamaForCausalLM, LogitsProcessorList
from transformers.generation.logits_process import SuppressTokensLogitsProcessor
import argparse
import json
import torch
from Utils import replicability
import tqdm
import random
import os


def text_parser(args, text):

    if "llama-zh" in args.pretained:
        output= text.split("系统是否应该在当前轮采取主动？")[-1].strip().replace('\n', '')
    else:
        output = text.split("Should the system take the initiative at the current turn?")[-1].strip().replace('\n', '')


    if output.startswith('yes') or output.startswith('是'):
        output='Initiative'
    elif output.startswith('no') or output.startswith('否'):
        output='Non-initiative'
    else:
        output = "None" if output=="" else output
        args.invalid_num+=1
        print(f"Invalid num {args.invalid_num}---------")
        print(output)

    return output

class Prompter:
    # following https://github.com/kyriemao/LLMCS/blob/master/promptor.py
    def __init__(self, args):
        self.args=args
        if "llama-zh" in self.args.pretained:
            self.instruction = "在一个多轮对话的场景下，给定当前轮的用户输入和之前轮的用户与系统的对话历史，预测系统在当前轮是否应该采取主动。请输出“是”或“否”。“是”意味着系统应该在当前轮采取主动，例如通过向用户询问澄清问题或者请求反馈；“否”意味着系统不应该在当前轮采取主动，例如通过向用户返回答案。"
        else:
            self.instruction = "Given the user utterance at current turn and the conversational history at previous turns, predict whether the system should take the initiative or not at the current turn. Please output \"yes\" or \"no\". \"yes\" means the system should take the initiative at the current turn by asking a clarifying question or requesting feedback and so on; \"no\" means the system should not take the initiative at the current turn, e.g., giving an answer to the user."

    def build_demo_prompt(self, conversations: list):
        demo_prompt = []
        demo_prompt.append(self.instruction)

        for conversation in conversations:
            conversation_prompt = []
            for turn_idx, turn in enumerate(conversation):
                turn_prompt = self.build_turn_prompt(turn_idx, turn, is_demo=True)
                conversation_prompt.append(turn_prompt)
            demo_prompt.append("\n".join(conversation_prompt))

        return "\n\n".join(demo_prompt)

    def build_turn_prompt(self, turn_idx, turn, is_demo):
        turn_prompt = []
        if "llama-zh" in self.args.pretained:
            turn_prompt.append("第{}轮".format(turn_idx + 1))
            turn_prompt.append("用户输入：{}".format(turn["user_utterance"]))
        else:
            turn_prompt.append("Turn: {}".format(turn_idx+1))
            turn_prompt.append("User utterance: {}".format(turn["user_utterance"]))

        if is_demo:
            if "llama-zh" in self.args.pretained:
                turn_prompt.append("系统是否应该在当前轮采取主动？{}".format("是" if turn["system_I_label"] == "Initiative" else "否"))
                turn_prompt.append("系统回复：{}".format(turn["system_utterance"]))
            else:
                turn_prompt.append("Should the system take the initiative at the current turn? {}".format("yes" if turn["system_I_label"] == "Initiative" else "no"))
                turn_prompt.append("System utterance: {}".format(turn["system_utterance"]))
        else:  # for test
            if "llama-zh" in self.args.pretained:
                turn_prompt.append("系统是否应该在当前轮采取主动？")
            else:
                turn_prompt.append("Should the system take the initiative at the current turn?")

        return "\n".join(turn_prompt)

    def build_this_turn_prompt_for_prediction(self, pre_prompt, this_turn_idx, this_turn, last_SIP, last_sys_utterance):
        pre_prompt_components = pre_prompt.split("\n\n")
        # update the last turn of the last dialog's info in the prompt
        if last_SIP is not None:
            last_conversation_prompt = pre_prompt_components[-1]
            pre_prompt_components.pop()
            last_conversation_prompt_turns = last_conversation_prompt.split('\n') # find the incomplete last turn
            if "llama-zh" in self.args.pretained:
                last_conversation_prompt_turns[-1] = "系统是否应该在当前轮采取主动？{}".format("是" if last_SIP=="Initiative" else "否")  # replace the empty "System initiative-taking decision" with the one with value
                last_conversation_prompt_turns.append("系统回复：{}".format(last_sys_utterance))
            else:
                last_conversation_prompt_turns[-1] = "Should the system take the initiative at the current turn? {}".format("yes" if last_SIP=="Initiative" else "no") # replace the empty "System initiative-taking decision" with the one with value
                last_conversation_prompt_turns.append("System utterance: {}".format(last_sys_utterance))
        else:
            last_conversation_prompt_turns = []

        this_turn_prompt = self.build_turn_prompt(this_turn_idx ,this_turn, is_demo=False)
        last_conversation_prompt_turns.append(this_turn_prompt)
        pre_prompt_components.append("\n".join(last_conversation_prompt_turns))

        return "\n\n".join(pre_prompt_components)


class LLM:
    def __init__(self, args):
        self.args=args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LlamaForCausalLM.from_pretrained(self.args.pretained, load_in_4bit=True, device_map="auto") # load_in_8bit=True,
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretained, padding_side="left")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.model.eval()

        print(f"Vocab of the model: {self.model.get_input_embeddings().weight.size(0)}")
        print(f"Vocab of the tokenizer: {len(self.tokenizer)}")

        self.args.invalid_num=0

        if "llama-zh" in self.args.pretained:
            save_token_id = self.tokenizer("是 \n 否", return_tensors='pt', padding='longest', truncation=True)['input_ids'][0]
        else:
            save_token_id = self.tokenizer("yes \n no", return_tensors='pt', padding='longest', truncation=True)['input_ids'][0]

        save_token_id = set(save_token_id.tolist())
        token_id = set([id for id in range(len(self.tokenizer))])
        suppress_token_id = token_id-save_token_id
        self.logits_processor_list = LogitsProcessorList([SuppressTokensLogitsProcessor(suppress_token_id)])

    def transform(self, examples):
        it = range(0, len(examples["example_id"]), self.args.batch_size)

        for start_idx in tqdm.tqdm(it):
            # one batch
            rng = slice(start_idx, start_idx + self.args.batch_size)

            enc = self.tokenizer(examples['input'][rng], return_tensors='pt', padding='longest', truncation=True)

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_new_tokens=self.args.max_new_tokens, # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                    logits_processor=self.logits_processor_list,
                )

            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for output in outputs:
                examples["output"].append(text_parser(self.args, output))

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='SIP')
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_path", type=str, default='')
    parser.add_argument("--pretained", type=str, default="")

    parser.add_argument("--input_path", type=str, default='')
    parser.add_argument("--demonstration_path", type=str, default='')
    parser.add_argument("--demonstration_num", type=int, default=4)

    parser.add_argument("--truncation_side", type=str, default='left')

    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=4)

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--interval", type=int, default=100)

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
    args.output_path = os.path.join(args.output_path, args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    promptor = Prompter(args)
    conversations = torch.load(args.demonstration_path)
    conv_idx_sampled = random.sample(range(len(conversations)), args.demonstration_num)

    print(f"conv_idx_sampled: {conv_idx_sampled}")

    conversations_sampled = [conversations[conv_idx] for conv_idx in conv_idx_sampled]
    demo_prompt = promptor.build_demo_prompt(conversations_sampled)

    examples={"example_id":[],"input":[],"output":[]}

    conversations = torch.load(args.input_path)
    for conv_idx, conversation in enumerate(conversations):

        pre_prompt = demo_prompt
        last_SIP, last_sys_utterance = None, None

        for turn_idx, turn in enumerate(conversation):
            #print("============one example=========")
            examples["example_id"].append(turn["example_id"])
            prompt = promptor.build_this_turn_prompt_for_prediction(pre_prompt, turn_idx, turn, last_SIP, last_sys_utterance)

            examples["input"].append(prompt)

            last_SIP = turn["system_I_label"]
            last_sys_utterance = turn["system_utterance"]
            pre_prompt = prompt

    llm = LLM(args)
    llm.transform(examples)

    with open(os.path.join(args.output_path, str(args.demonstration_num) + ".txt"), 'w') as w:
        for idx, example_id in enumerate(examples["example_id"]):
            w.write(example_id + '\t' + str(examples["output"][idx]) + '\n')
