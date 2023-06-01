from model.Utils import universal_sentence_embedding
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import BertModel
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class utterance_encoding(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args= args
        if args.dataset == "WISE":
            self.enc = BertModel.from_pretrained('bert-base-chinese')
        elif args.dataset == "MSDialog" or "ClariQ":
            self.enc = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data):
        batch_size, conversation_len, max_utterance_len = data['user_utterance'].size()

        user_utterance = data['user_utterance'] # [batch, ?, max_utterance_len]
        user_utterance = user_utterance.reshape(-1, max_utterance_len)  # [batch * ?, max_utterance_len]
        user_utterance_mask = user_utterance.ne(0).detach() # [batch * ?, max_utterance_len]

        system_utterance =  data['system_utterance'] # [batch, ?, max_utterance_len]
        system_utterance = system_utterance.reshape(-1, max_utterance_len)  # [batch * ?, max_utterance_len]
        system_utterance_mask = system_utterance.ne(0).detach() # [batch * ?, max_utterance_len]
        
        encoded_user_utterance = self.enc(user_utterance, attention_mask=user_utterance_mask.float())[0]  # [batch * ?, max_utterance_len, hidden_size]
        pooling_user_utterance = universal_sentence_embedding(encoded_user_utterance, user_utterance_mask) # [batch * ?, hidden_size]
        pooling_user_utterance /= np.sqrt(pooling_user_utterance.size()[-1]) # # [batch * ?, hidden_size]

        encoded_system_utterance = self.enc(system_utterance, attention_mask=system_utterance_mask.float())[0]  # [batch * ?, max_utterance_len, hidden_size]
        pooling_system_utterance = universal_sentence_embedding(encoded_system_utterance, system_utterance_mask) # [batch * ?, hidden_size]
        pooling_system_utterance /= np.sqrt(pooling_system_utterance.size()[-1]) # [batch * ?, hidden_size]
        # [batch, ?, hidden_size]
        return pooling_user_utterance.reshape(batch_size, conversation_len, -1), pooling_system_utterance.reshape(batch_size, conversation_len, -1)

class posterior_conversation_encoding(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args=args
        self.lstm = nn.LSTM(self.args.hidden_size, self.args.hidden_size, dropout= self.args.dropout, num_layers=self.args.BiLSTM_layers, bidirectional=True, batch_first=True)
        
    def forward(self, input):
        output,(_,_) = self.lstm(input)  # [batch_size, ?, 2*hidden_size]
        return output  # [batch_size, ?, 2]

class prior_conversation_encoding(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args=args
        self.lstm = nn.LSTM(self.args.hidden_size, self.args.hidden_size, dropout= self.args.dropout, num_layers=self.args.BiLSTM_layers, bidirectional=True, batch_first=True)
    def forward(self, input):
        output,(_,_) = self.lstm(input)  # [batch_size, ?, 2*hidden_size]
        return output  # [batch_size, ?, 2]

class crf(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args=args

        self.matrice_all = nn.Parameter(torch.Tensor(2, 2))

        # role
        self.matrice_u2s = nn.Parameter(torch.Tensor(2, 2))
        self.matrice_s2u = nn.Parameter(torch.Tensor(2, 2))

        # the number of times of initiative-taking of the system
        self.matrice_u2s_I0 = nn.Parameter(torch.Tensor(2, 2))  # no initiave-taking before
        self.matrice_u2s_I1 = nn.Parameter(torch.Tensor(2, 2))  # 1 system's initiave-taking before
        self.matrice_u2s_I2_more = nn.Parameter(torch.Tensor(2, 2))  # 2 and more

        # the distance from the last system's initiative-taking to the current system's turn
        self.matrice_u2s_Id2 = nn.Parameter(torch.Tensor(2, 2))  # consecutive  e.g., 4-2=2
        self.matrice_u2s_Id4_more = nn.Parameter(torch.Tensor(2, 2))  # inconsecutive e.g., 6-2=4 and  8-2=6

        self.matrice_1_2 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_2_3 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_3_4 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_4_5 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_5_6 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_6_7 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_7_8 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_8_9 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_9_10 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_10_11 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_11_12 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_12_13 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_13_14 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_14_15 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_15_16 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_16_17 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_17_18 = nn.Parameter(torch.Tensor(2, 2))  # u2s
        self.matrice_18_19 = nn.Parameter(torch.Tensor(2, 2))  # s2u
        self.matrice_19_20 = nn.Parameter(torch.Tensor(2, 2))  # u2s
       
    def forward(self, emission_scores, label, prior, posterior_sequence, state):
        # the input and output all don't have bath dimension
        # emission_scores [?, 2]  should be posterior sequence
        # label [?]
        # prior_sequence [?-1, 2*hidden_size_BiLSTM]
        # posterior_sequence [?, 2*hidden_size_BiLSTM]
        # state [?]

        conversation_len = emission_scores.shape[0]

        assert emission_scores.shape[0] == label.shape[0] 
        assert conversation_len % 2 == 0

        if self.args.mode=="train":
            assert posterior_sequence.shape[0] % 2==0
        elif self.args.mode == "inference":
            assert posterior_sequence.shape[0] % 2==1
            
        padding_matrice = torch.zeros(2, 2).cuda()
        
        who2who_subsidiary = torch.stack([self.matrice_u2s, self.matrice_s2u, padding_matrice],0)  # [3, 2, 2]
        Intime_subsidiary = torch.stack([self.matrice_u2s_I0, self.matrice_u2s_I1, self.matrice_u2s_I2_more, padding_matrice], 0) # [4, 2, 2]
        Distance_subsidiary = torch.stack([self.matrice_u2s_Id2, self.matrice_u2s_Id4_more, padding_matrice], 0) # [3, 2, 2]
        position_subsidiary = torch.stack( # [18, 2, 2]
            [self.matrice_1_2,
             self.matrice_2_3,
             self.matrice_3_4,
             self.matrice_4_5,
             self.matrice_5_6,
             self.matrice_6_7,
             self.matrice_7_8,
             self.matrice_8_9,
             self.matrice_9_10,
             self.matrice_10_11,
             self.matrice_11_12,
             self.matrice_12_13,
             self.matrice_13_14,
             self.matrice_14_15,
             self.matrice_15_16,
             self.matrice_16_17,
             self.matrice_17_18,
             self.matrice_18_19,
             self.matrice_19_20,
             padding_matrice],
            0)
        overall_subsidiary = torch.stack([self.matrice_all, padding_matrice],0)  # [2, 2, 2]
        bank = [who2who_subsidiary, position_subsidiary, Intime_subsidiary, Distance_subsidiary, overall_subsidiary]  # [4, x, 2, 2]

        if self.args.mode=="train":
            # obtain golden socre
            front_pointers= label[:-1]  # remove the last label
            back_pointers = label[1:]  # remove the first label
            assert front_pointers.shape[0]==back_pointers.shape[0]==conversation_len-1

            sum_emission_score = torch.sum(emission_scores[range(conversation_len), label])

            sum_transition_score =0
            for index in range(conversation_len):
                if index > 0: # sanity check
                    if self.args.model == "VanillaCRF":
                        combined_matrice = bank[4][state["overall"][index]]
                    elif self.args.model == "Who2whoCRF":
                        combined_matrice = bank[0][state["who2who"][index]]
                    elif self.args.model == "PositionCRF":
                        combined_matrice = bank[1][state["position"][index]]
                    elif self.args.model == "Who2who_PositionCRF":
                        combined_matrice = bank[0][state["who2who"][index]]+bank[1][state["position"][index]]
                    elif self.args.model == "IntimeCRF":
                        if state["who2who"][index]==1:
                            assert state["Intime"][index]==-1
                            combined_matrice = bank[0][1] # syetm2user
                        else:
                            combined_matrice = bank[2][state["Intime"][index]]
                    elif self.args.model == "DistanceCRF":
                        if state["who2who"][index] == 1:
                            assert state["Distance"][index] == -1
                            combined_matrice = bank[0][1]  # syetm2user
                        elif state["Intime"][index] == 0:
                            combined_matrice = bank[2][0]
                            assert state["Distance"][index] == -1
                        else:
                            combined_matrice = bank[3][state["Distance"][index]]

                    sum_transition_score += combined_matrice[front_pointers[index-1], back_pointers[index-1]] # select right values from the matrice

            gold_score = sum_emission_score + sum_transition_score

            # obtain total socre
            alpha = torch.full((1, 2), 0.0, device=emission_scores.device) # [1, 2]
            for index in range(conversation_len):
                # [[p1, p1] + [[e1, e2]  + [[t11, t12]
                #  [p2, p2]]   [e1, e2]]    [t21, t22]]   [2, 2]

                if self.args.model == "VanillaCRF":
                    combined_matrice = bank[4][state["overall"][index]]
                elif self.args.model == "Who2whoCRF":
                    combined_matrice = bank[0][state["who2who"][index]]
                elif self.args.model == "PositionCRF":
                    combined_matrice = bank[1][state["position"][index]]
                elif self.args.model == "Who2who_PositionCRF":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[1][state["position"][index]]
                elif self.args.model == "IntimeCRF":
                    if state["who2who"][index] == 1:
                        assert state["Intime"][index] == -1
                        combined_matrice = bank[0][1]  # syetm2user
                    else:
                        combined_matrice = bank[2][state["Intime"][index]]
                elif self.args.model == "DistanceCRF":
                    if state["who2who"][index] == 1:
                        assert state["Distance"][index] == -1
                        combined_matrice = bank[0][1]  # syetm2user
                    elif state["Intime"][index] == 0:
                        combined_matrice = bank[2][0]
                        assert state["Distance"][index] == -1
                    else:
                        combined_matrice = bank[3][state["Distance"][index]]
                if index ==0:
                    assert torch.equal(combined_matrice, torch.zeros(2,2).int().cuda())

                alpha = torch.logsumexp(alpha.T + emission_scores[index].unsqueeze(0) + combined_matrice, dim=0, keepdim=True)  # [1, 2]  row vector

            total_score = torch.logsumexp(alpha.T, dim=0).squeeze()

            return gold_score, total_score

        elif self.args.mode=="inference":

            backtrace = []
            alpha = torch.full((1, 2), 0.0, device= emission_scores.device) # [1, 2]  row vector

            for index in range(conversation_len):
                # [[p1, p1] + [[e1, e2]  + [[t11, t12]
                #  [p2, p2]]   [e1, e2]]    [t21, t22]]
                if self.args.model == "VanillaCRF":
                    combined_matrice = bank[4][state["overall"][index]]
                elif self.args.model == "Who2whoCRF":
                    combined_matrice = bank[0][state["who2who"][index]]
                elif self.args.model == "PositionCRF":
                    combined_matrice = bank[1][state["position"][index]]
                elif self.args.model == "Who2who_PositionCRF":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[1][state["position"][index]]
                elif self.args.model == "IntimeCRF":
                    if state["who2who"][index] == 1:
                        assert state["Intime"][index] == -1
                        combined_matrice = bank[0][1]  # syetm2user
                    else:
                        combined_matrice = bank[2][state["Intime"][index]]
                elif self.args.model == "DistanceCRF":
                    if state["who2who"][index] == 1:
                        assert state["Distance"][index] == -1
                        combined_matrice = bank[0][1]  # syetm2user
                    elif state["Intime"][index] == 0:
                        combined_matrice = bank[2][0]
                        assert state["Distance"][index] == -1
                    else:
                        combined_matrice = bank[3][state["Distance"][index]]

                if index == 0:
                    assert torch.equal(combined_matrice, torch.zeros(2, 2).int().cuda())


                alpha = alpha.T + emission_scores[index].unsqueeze(0) + combined_matrice  # [2, 2]

                viterbivars_t, bptrs_t = torch.max(alpha, dim=0) # [2], [2]

                backtrace.append(bptrs_t)
                alpha = viterbivars_t.unsqueeze(0)  # [1, 2]  row vector

            # backtrack
            best_tag_id = alpha.flatten().argmax().item()
            best_path = [best_tag_id]

            assert torch.equal(backtrace[0], torch.zeros(2).int().cuda())

            for bptrs_t in reversed(backtrace[1:]):  # ignore the first one
                best_tag_id = bptrs_t[best_tag_id].item()
                best_path.append(best_tag_id)

            best_path.reverse()

            assert len(best_path) % 2 == 0
            assert len(best_path) == conversation_len

            return best_path


class BILSTMCRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.utterance_encoding=utterance_encoding(args=args)
        self.posterior_conversation_encoding = posterior_conversation_encoding(args=args)
        self.prior_conversation_encoding = prior_conversation_encoding(args=args)
        self.crf = crf(args=args)
        self.prior_e_project = nn.Linear(2 * self.args.hidden_size, 2)
        self.posterior_e_project = nn.Linear(2 * self.args.hidden_size, 2)

    def forward(self, data):
        # pooling_user_utterance [batch=1, ?, hidden_size]
        # pooling_system_utterance [batch=1, ?, hidden_size]
        pooling_user_utterance, pooling_system_utterance = self.utterance_encoding(data)
        batch_size, pair_num, hidden_size = pooling_user_utterance.size()

        previous_utterance_sequence = []
        previous_I_label_sequence = []

        logger ={"role":[], "system_I":[]}

        I_label_sequence_batch = []

        if self.args.mode == 'train':
            prior_emission_score_batch = []
            posterior_emission_score_batch = []

            gold_score_batch = []
            total_score_batch = []

        elif self.args.mode == 'inference':
            predicted_path_batch = []
            predicted_path_batch_from_emission = []

        # traverse all turns (user-system pairs) in a conversation
        for i in range(pair_num):
            # batch size is always one
            previous_utterance_sequence.append(pooling_user_utterance[:, i, :].unsqueeze(1))  # add user's utterance [1, 1, hidden_size]
            prior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+1, hidden_size]
            assert prior_utterance_sequence.shape[1]==2*i+1

            logger["role"].append("user")
            logger["system_I"].append(0)

            previous_utterance_sequence.append(pooling_system_utterance[:, i, :].unsqueeze(1))  # add corresponding system's utterance [1, 1, hidden_size]
            posterior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+2, hidden_size]
            assert posterior_utterance_sequence.shape[1] == 2*(i+1)

            logger["role"].append("system")
            logger["system_I"].append(data['system_I_label'][:, i].squeeze().item())  # add 1 or 0

            assert len(logger["role"]) == len(logger["system_I"]) == 2 * (i + 1)

            previous_I_label_sequence.append(data['user_I_label'][:, i].unsqueeze(1))  # add user's utterance I label [1, 1]
            previous_I_label_sequence.append(data['system_I_label'][:, i].unsqueeze(1))  # add system's utterance I label [1,1]
            I_label_sequence = torch.cat(previous_I_label_sequence, 1)  # [1, 2i+2]
            assert I_label_sequence.shape[1] == 2 * (i + 1)

            I_label_sequence_batch.append(I_label_sequence.squeeze().tolist())  # [[2], [4], ...] only used for inference

            state = {"who2who": [], "position": [], "Intime": [], "Distance": [], "overall": []}

            for turn_index in range(len(logger["role"])):
                if turn_index == 0:
                    state["who2who"].append(-1)
                    state["position"].append(-1)
                    state["Intime"].append(-1)
                    state["Distance"].append(-1)
                    state["overall"].append(-1)
                else:
                    state["overall"].append(0)

                    if turn_index <= 19:
                        state["position"].append(turn_index - 1)
                    else:
                        state["position"].append(-1)

                    if logger["role"][turn_index - 1] == "system":
                        # we don't concentrate on this perspective
                        state["who2who"].append(1)
                        state["Intime"].append(-1)
                        state["Distance"].append(-1)

                    else:
                        # user to system | the last one is user
                        state["who2who"].append(0)
                        I_times = sum(logger["system_I"][0:turn_index - 1])
                        if I_times == 0:
                            state["Intime"].append(0)
                            state["Distance"].append(-1)
                        else:
                            if I_times == 1:
                                state["Intime"].append(1)
                            else:
                                state["Intime"].append(1)

                            last_system_I_turn = -1
                            for turn_index_ in range(len(logger["system_I"][0:turn_index - 1])):
                                if logger["system_I"][turn_index_] == 1:
                                    last_system_I_turn = turn_index_

                            distance = turn_index - last_system_I_turn

                            if distance == 2:
                                state["Distance"].append(0)
                            else:
                                state["Distance"].append(1)

            assert len(state["Distance"]) == len(state["Intime"]) == len(state["who2who"]) == len(
                state["position"]) == len(state["overall"]) == len(logger["role"]) == len(logger["system_I"])

            prior_hidden_squence = self.prior_conversation_encoding(prior_utterance_sequence)
            prior_emission_score = self.prior_e_project(prior_hidden_squence[:, -1, :])

            if self.args.mode == 'train':
                # obtain emission score
                posterior_hidden_squence = self.posterior_conversation_encoding(posterior_utterance_sequence)
                posterior_emission_scores = self.posterior_e_project(posterior_hidden_squence)

                gold_score, total_score= self.crf(posterior_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), posterior_hidden_squence.squeeze(0), state)

                gold_score_batch.append(gold_score.unsqueeze(0))
                total_score_batch.append(total_score.unsqueeze(0))

                prior_emission_score_batch.append(prior_emission_score.unsqueeze(1))
                posterior_emission_score_batch.append(posterior_emission_scores[:, -1, :].unsqueeze(1))

        # End of training cycle
            elif self.args.mode == 'inference':
                partial_posterior_hidden_squence = self.posterior_conversation_encoding(prior_utterance_sequence)  #  [1, 2i+1, 2*hidden_size_BiLSTM]
                partial_posterior_emission_scores = self.posterior_e_project(partial_posterior_hidden_squence)  # [1, 2i+1, 2]

                combined_emission_scores = torch.cat([partial_posterior_emission_scores, prior_emission_score.unsqueeze(1)], 1)  # [1, 2i+2, 2]
                predicted_path = self.crf(combined_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), partial_posterior_hidden_squence.squeeze(0) , state)
                assert len(predicted_path) == combined_emission_scores.shape[1] == (partial_posterior_emission_scores.shape[1] + 1) == (partial_posterior_hidden_squence.shape[1]+1) == (prior_hidden_squence.shape[1]+1)
                predicted_path_batch.append(predicted_path) # [[2], [4], ...]
                predicted_path_batch_from_emission.append(combined_emission_scores.squeeze(0).max(1)[1].tolist())  # [1, 2i+2, 2] --> [2i+2, 2] -->[2i+2]

        if self.args.mode == 'train':
            gold_score_tensor = torch.cat(gold_score_batch)  # [pair_num]
            total_score_tensor = torch.cat(total_score_batch)  # [pair_num]

            prior_emission_score_tensor = torch.cat(prior_emission_score_batch, 1).squeeze(0)  # [pair_num, 2]
            posterior_emission_score_tensor = torch.cat(posterior_emission_score_batch, 1).squeeze(0)  # [pair_num,2]

            assert pair_num == prior_emission_score_tensor.shape[0] == posterior_emission_score_tensor.shape[0]
            assert prior_emission_score_tensor.shape[1] == posterior_emission_score_tensor.shape[1] # 2

            loss_crf = torch.mean(total_score_tensor - gold_score_tensor) # average each sample
            loss_mle_e = F.mse_loss(prior_emission_score_tensor, posterior_emission_score_tensor.detach())  # [pair_num, 2]

            return {"loss_crf": loss_crf, "loss_mle_e": loss_mle_e}

        elif self.args.mode == 'inference':
            assert len(predicted_path_batch[0])==len(predicted_path_batch_from_emission[0])==len(I_label_sequence_batch[0])==2
            
            if len(predicted_path_batch)>1:
                assert len(predicted_path_batch[1])==len(predicted_path_batch_from_emission[1])==len(I_label_sequence_batch[1])==4

            return predicted_path_batch

class ContextEncoding(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        if args.dataset == "WISE":
            self.enc = BertModel.from_pretrained('bert-base-chinese')
        elif args.dataset == "MSDialog":
            self.enc = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data):
        batch_size, pair_num, max_context_len = data['context'].size()
        context = data['context']  # [batch, ?, max_context_len]
        context = context.reshape(-1, max_context_len)  # [batch * ?, max_context_len]
        context_mask = context.ne(0).detach()  # [batch * ?, max_context_len]
        encoded_context = self.enc(context, attention_mask=context_mask.float())[0]  # [batch * ?, max_context_len, hidden_size]

        return encoded_context[:, 0, :].reshape(batch_size, pair_num, -1)  # [CLS] [batch, ?, hidden_size]

class OneStep(nn.Module):
    def __init__(self, args=None, actiom_embedding=None):
        super().__init__()
        self.args = args
        self.gru = nn.GRU(self.args.hidden_size*2, self.args.hidden_size, bidirectional=False)
        self.action_embedding = actiom_embedding

    def forward(self, tgt, state, feature):
        # single step
        # tgt [pari_num, 1]
        # state [pari_num, 1, hidden_size]
        # feature [pari_num, 1, hidden_size]
        embedded = self.action_embedding(tgt)  # [pari_num, 1, hidden_size]
        input = torch.cat([embedded, feature], 2)  # [pari_num, 1, 2*hidden_size]
        output, state = self.gru(input.transpose(0, 1), state.transpose(0, 1))
        output = output.transpose(0, 1)  # [pari_num, 1, hidden_size]
        state = state.transpose(0, 1)  # [pari_num, 1, hidden_size]
        return output, state

class ActionPrediction(nn.Module):
    def __init__(self, args, config,mlb):
        super().__init__()
        self.args = args
        self.config = config
        self.mlb = mlb

        self.context_encoding = ContextEncoding(args=args)

        if args.model == 'mlc':
            if args.dataset == "WISE":
                action_num = 23
            elif args.dataset == "MSDialog":
                action_num = 12

            if args.task=="SIP-AP":
                self.d_embedding = nn.Embedding(2, self.args.hidden_size)
                self.map_state2action = nn.Linear(self.args.hidden_size*2, action_num)
            elif args.task=="AP":
                self.map_state2action = nn.Linear(self.args.hidden_size, action_num)
            else:
                raise NotImplementedError

        elif args.model == 'sg':
            if args.dataset == "WISE":
                action_num = 26  # 23+3
            elif args.dataset == "MSDialog":
                action_num = 15  # 12+3

            if args.task=="SIP-AP":
                self.d_embedding = nn.Embedding(2, self.args.hidden_size)
                self.map_enc2dec = nn.Linear(self.args.hidden_size*2, self.args.hidden_size)
            elif args.task == "AP":
                self.map_enc2dec = nn.Linear(self.args.hidden_size, self.args.hidden_size)
            else:
                raise NotImplementedError

            self.map_state2action = nn.Linear(self.args.hidden_size, action_num)
            self.action_embedding = nn.Embedding(action_num, self.args.hidden_size, padding_idx=0)
            self.dec = OneStep(args=args, actiom_embedding=self.action_embedding)

    def forward(self, data):
        # batch size is always one
        cls = self.context_encoding(data)  # [CLS] [batch_size, pair_num, hidden_size]
        batch_size, pair_num, hidden_size = cls.size()

        if self.args.model == 'mlc':
            if self.args.task=="SIP-AP":
                if self.args.mode == "inference" and self.args.Oracle_SIP==False:
                    d = self.d_embedding(data['system_I_prediction'])  # [batch_size, pair_num, 768]
                else:
                    d = self.d_embedding(data['system_I_label'])  # [batch_size, pair_num, 768]
                logits = self.map_state2action(torch.cat([cls, d], 2))  # [batch=1, pair_num, 2*768]-->[batch_size, pair_num, 23/12]
            elif self.args.task == "AP":
                logits = self.map_state2action(cls)  # [batch_size, pair_num, 768]-->[batch_size, pair_num, 23/12]
            else:
                raise NotImplementedError

            if self.args.mode == 'train':
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(0),data['system_action_label'].float().squeeze(0).detach()).unsqueeze(0)
                return {"loss": loss}
            elif self.args.mode == 'inference':
                predicted = torch.where(torch.sigmoid(logits) > 0.5, 1, 0)  # [batch=1, pair_num, 23/12]
                predicted_mlb = self.mlb.inverse_transform(predicted.squeeze(0).cpu())
                for idx, row in enumerate(predicted_mlb):
                    predicted_mlb[idx]=list(row)

                predicted_verfied=[]
                for row in predicted.squeeze(0):
                    predicted_verfied.append([self.config.action[idx] for idx, col in enumerate(row) if col>0])


                assert predicted_mlb==predicted_verfied
                return predicted_mlb

        elif self.args.model == 'sg':
            if self.args.task=="SIP-AP":
                if self.args.mode == "inference" and self.args.Oracle_SIP==False:
                    d = self.d_embedding(data['system_I_prediction'])  # [batch=1, ?, 768]
                else:
                    d = self.d_embedding(data['system_I_label'])  # [batch=1, ?, 768]
                # [batch, pair_num, 2*768]-->[batch, pair_num, 768] --> [pair_num, hidden_size]--> [pair_num, 1, hidden_size]
                initialised_decoder_state = self.map_enc2dec(torch.cat([cls, d], 2)).squeeze(0).unsqueeze(1)
            elif self.args.task == "AP":
                # [batch, pair_num, 768]-->[batch, pair_num, 768] --> [pair_num, 768]--> [pair_num, 1, 768]
                initialised_decoder_state = self.map_enc2dec(cls).squeeze(0).unsqueeze(1)
            else:
                raise NotImplementedError

            system_action_sequence = data['system_action_sequence'].squeeze(0)  # [batch_size, pair_num, max_target_length]-->[pair_num, max_target_length]
            pair_num, max_target_length = system_action_sequence.size()  # [pair_num, max_target_length]

            if self.args.mode == 'train':
                token_index = torch.Tensor([1] * pair_num).unsqueeze(1).long().cuda()  # [pair_num, 1]

                outputs_on_actions = []
                states = [initialised_decoder_state]  # [pair_num, 1, hidden_size]

                for t in range(max_target_length):
                    output, state = self.dec(token_index, states[-1], initialised_decoder_state)
                    # output [pair_num, 1, hidden_size]
                    # state  [pair_num, 1, hidden_size]
                    output_on_action = self.map_state2action(output)  # [pair_num, 1, label_size]
                    states.append(state)
                    outputs_on_actions.append(output_on_action)
                    token_index = system_action_sequence[:, t].unsqueeze(1)  # [pair_num, 1]

                assert (len(states) - 1) == len(outputs_on_actions) == max_target_length

                outputs_on_actions = torch.cat(outputs_on_actions, dim=1)  # [pair_num, max_target_length, label_size]

                loss = F.cross_entropy(outputs_on_actions.view(-1, outputs_on_actions.size(-1)),data['system_action_sequence'].view(-1), ignore_index=0)
                return {"loss": loss}

            elif self.args.mode == 'inference':
                greedy_indices = []
                greedy_ends = torch.Tensor([0] * pair_num).unsqueeze(1).long().cuda() == 1  # [pair_num, 1] all values are false

                token_index = torch.Tensor([1] * pair_num).unsqueeze(1).long().cuda()  # [pair_num, 1]
                states = [initialised_decoder_state]  # [pair_num, 1, hidden_size]

                for t in range(max_target_length):
                    output, state = self.dec(token_index, states[-1], initialised_decoder_state)
                    # output [pair_num, 1, hidden_size]
                    # state  [pair_num, 1, hidden_size]
                    states.append(state)
                    output_on_action = self.map_state2action(output)  # [pair_num, 1, label_size]
                    # probs [pair_num, 1]
                    # id [pair_num, 1]
                    probs, id = torch.max(output_on_action, dim=2)

                    greedy_end = id == 2  # [pair_num, 1]
                    id.masked_fill_(greedy_ends, 0)  # [pair_num, 1]
                    greedy_indices.append(id)  # [[pair_num, 1], [pair_num, 1],....]

                    greedy_ends = greedy_ends | greedy_end  # [pair_num, 1]
                    token_index = id  # [pair_num, 1]

                predicted = torch.cat(greedy_indices, dim=1)  # [pair_num, max_target_length]

                actions_pairs = []
                for pair in predicted:
                    action_pair = []
                    for id in pair:
                        action = self.config.id2action[id.item()]
                        if action == "[PAD]" or action == "[SOA]":
                            continue
                        if action == "[EOA]":
                            break
                        action_pair.append(action)
                    if len(action_pair) == 0:
                        pass
                    actions_pairs.append(action_pair)
                return actions_pairs










