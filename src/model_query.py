from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import numpy as np
import re


class QueryModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim,
        dump,
        ds,
        L=30,
        device='cpu'
    ):
        super(QueryModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        self.tokenizer = tokenizer
        
        self.dump = dump
        self.model_start = dump.model_start
        self.model_end = dump.model_end

        self.H = torch.Tensor(dump.H).to(device)
        self.H.requires_grad = False
        
        self.ds = ds
        self.L = L
        self.device = device
        self.k = 100

    def forward(self, ids):
        N = len(ids)                    
        questions = self.ds.get_questions(ids)
        
        input_ids = self.tokenizer(questions, truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        last_hidden_state_start = self.model_start(**input_ids).last_hidden_state[:, 0, :]
        last_hidden_state_end = self.model_end(**input_ids).last_hidden_state[:, 0, :]
        print(last_hidden_state_start)

        dot_start = torch.matmul(self.H, last_hidden_state_start.T).T
        dot_end = torch.matmul(self.H, last_hidden_state_end.T).T


        for n in range(N):


            I_start = np.argsort(-dot_start[n].cpu().detach().numpy())[:self.k]
            I_end = np.argsort(-dot_end[n].cpu().detach().numpy())[:self.k]

            S_start = dot_start[n][I_start]
            S_end = dot_end[n][I_end]
            loss = torch.Tensor([0.]).to(self.device)

            answer_candidate2cumscore = {}
            answer_2cumscore = {}
            for s_start, token_w_id_start in zip(S_start, I_start):
                for s_end, token_w_id_end in zip(S_end, I_end):
                    context_id_candidate_start = self.dump.token_w_id2context_id[token_w_id_start]
                    context_id_candidate_end = self.dump.token_w_id2context_id[token_w_id_end]
                    if context_id_candidate_start == context_id_candidate_end:
                        
                        token_id_start = self.dump.token_w_id2token_id[token_w_id_start]
                        token_id_end = self.dump.token_w_id2token_id[token_w_id_end]
                        
                        answer_id = self.dump.context_id2id[context_id_candidate_start]
                        context = self.ds.contexts[answer_id]
                        context_ids = self.tokenizer(context)['input_ids']
                        
                        
                                
                        if token_id_start in context_ids and token_id_end in context_ids:
                            
                            start_index = context_ids.index(token_id_start)
                            end_index = context_ids.index(token_id_end)
                            answer_candidate_ids = context_ids[start_index:end_index]
                            answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                            if start_index <= end_index <= start_index + self.L:
                                if abs(start_index - self.ds.spans_input_ids[answer_id]['start']) / len(context_ids) < 0.5 and \
                                    abs(end_index - self.ds.spans_input_ids[answer_id]['end']) / len(context_ids) < 0.5:
                                    answer_2cumscore[(start_index, end_index, answer_candidate)] = s_start + s_end
                                
                                answer_candidate2cumscore[(start_index, end_index, answer_candidate)] = s_start + s_end   
            # print(answer_2cumscore)

            if answer_2cumscore:
                scores_numenator = torch.vstack(tuple(answer_2cumscore.values()))
            else:
                scores_numenator = torch.Tensor([0.], )
            if answer_candidate2cumscore:
                scores_denominator = torch.vstack(tuple(answer_candidate2cumscore.values()))
            else:
                scores_denominator = torch.Tensor([0.])
            numenator = torch.sum(torch.exp(scores_numenator))
            denominator = torch.sum(torch.exp(scores_denominator))
            loss += - torch.log(numenator / denominator)
                
        return loss

    def predict(self, ids, k=100, verbose=False, L=30):
        N = len(ids)
        questions = self.ds.get_questions(ids)
        contexts = self.ds.get_contexts(ids)
        if verbose:
            print(f"Q: {questions}")
            print(f"C: {contexts}")

        input_ids = self.tokenizer(questions, truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        last_hidden_state_start = self.model_start(**input_ids).last_hidden_state.detach().cpu().numpy()[:, 0, :].reshape((N, self.hidden_dim))
        last_hidden_state_end = self.model_end(**input_ids).last_hidden_state.detach().cpu().numpy()[:, 0, :].reshape((N, self.hidden_dim))
        S_start, I_start = self.dump.index.search(last_hidden_state_start, k)
        S_end, I_end = self.dump.index.search(last_hidden_state_end, k)
        
        
        answers, start_indices, end_indices, scores = [], [], [], []
        for n in range(N):
            answer_candidate2cumscore = {}
            for s_start, token_w_id_start in zip(S_start[n], I_start[n]):
                for s_end, token_w_id_end in zip(S_end[n], I_end[n]):
                    context_id_candidate_start = self.dump.token_w_id2context_id[token_w_id_start]
                    context_id_candidate_end = self.dump.token_w_id2context_id[token_w_id_end]
                    if context_id_candidate_start == context_id_candidate_end:
                        
                        token_id_start = self.dump.token_w_id2token_id[token_w_id_start]
                        token_id_end = self.dump.token_w_id2token_id[token_w_id_end]
                        
                        context = self.ds.contexts[self.dump.context_id2id[context_id_candidate_start]]
                        context_ids = self.tokenizer(context)['input_ids']
                        
                        if token_id_start in context_ids and token_id_end in context_ids:
                            start_index = context_ids.index(token_id_start)
                            end_index = context_ids.index(token_id_end)
                            if start_index <= end_index <= start_index + L:
                                answer_candidate_ids = context_ids[start_index:end_index]
                                answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                                answer_candidate2cumscore[(start_index, end_index, answer_candidate)] = s_start + s_end
                                
                        
            if answer_candidate2cumscore:
                answer, score = sorted(answer_candidate2cumscore.items(), key=lambda x: -x[1])[0]
                start_index, end_index, answer = answer
                answers.append(re.sub(r'#', '', answer))
                start_indices.append(start_index)
                end_indices.append(end_index)
                scores.append(score)
        return answers, start_indices, end_indices, scores