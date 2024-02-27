from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import numpy as np
import re
from typing import List


class QueryModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim,
        dump,
        L=30,
        device='cpu',
        k=10,
        p=0.3,
        strict=False
    ):
        super(QueryModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.p = p
        self.strict = strict
        self.tokenizer = tokenizer
        
        self.dump = dump
        self.model_start = dump.model_start
        self.model_end = dump.model_end

        self.H = torch.Tensor(dump.H).to(device)
        
        self.dataset = dump.dataset
        self.L = L
        self.device = device
        self.k = k
        self.penality = torch.Tensor([10e2]).to(device)
        self.penality.requires_grad = True

    def forward(self, question_ids: torch.Tensor, **kwargs):
                            
        
        last_hidden_state_start = self.model_start(**question_ids).last_hidden_state[:, 0, :]
        last_hidden_state_end = self.model_end(**question_ids).last_hidden_state[:, 0, :]
        dot_start = torch.matmul(self.H, last_hidden_state_start.T).T
        dot_end = torch.matmul(self.H, last_hidden_state_end.T).T
        N, _ = dot_start.shape
 
        
        loss = torch.Tensor([0.]).to(self.device)
        for n in range(N):

            
            I_start = np.argsort(-dot_start[n].cpu().detach().numpy())[:self.k]
            I_end = np.argsort(-dot_end[n].cpu().detach().numpy())[:self.k]

            S_start = dot_start[n][I_start]
            S_end = dot_end[n][I_end]
            

            candidate_cumscores = []
            cumscores = []
            for s_start, token_w_id_start in zip(S_start, I_start):
                for s_end, token_w_id_end in zip(S_end, I_end):
                    context_id_candidate_start = self.dump.token_w_id2context_id[token_w_id_start]
                    context_id_candidate_end = self.dump.token_w_id2context_id[token_w_id_end]
                    candidate_cumscores.append((s_start + s_end) / self.hidden_dim)
                    if context_id_candidate_start == context_id_candidate_end:
                        
                        token_id_start = self.dump.token_w_id2token_id[token_w_id_start]
                        token_id_end = self.dump.token_w_id2token_id[token_w_id_end]
                        
                        idx = self.dump.context_id2id[context_id_candidate_start]
                        context = self.dataset.contexts[idx]
                        context_ids = self.tokenizer(context)['input_ids']
                        
                                
                        if token_id_start in context_ids and token_id_end in context_ids:
                            
                            start_index = context_ids.index(token_id_start)
                            end_index = context_ids.index(token_id_end)
                            answer_candidate_ids = context_ids[start_index:end_index]
                            if start_index <= end_index <= start_index + self.L:
                                if not self.strict:
                                  if abs(start_index - self.dataset.spans_input_ids[idx]['start']) / len(context_ids) < self.p and \
                                      abs(end_index - self.dataset.spans_input_ids[idx]['end']) / len(context_ids) < self.p:
                                    cumscores.append((s_start + s_end) / self.hidden_dim )
                                else:
                                  if start_index == self.dataset.spans_input_ids[idx]['start'] and \
                                    end_index == self.dataset.spans_input_ids[idx]['end']:
                                    cumscores.append((s_start + s_end) / self.hidden_dim )
                                    
                      
            # print(answer_2cumscore)
            
            if candidate_cumscores and cumscores:
                scores_numenator = torch.vstack(tuple(cumscores))
                scores_denominator = torch.vstack(tuple(candidate_cumscores))
                print(len(scores_numenator), len(scores_denominator))
                # print(scores_numenator, scores_denominator)
                scores_numenator = scores_numenator - torch.max(scores_denominator)
                scores_denominator = scores_denominator - torch.max(scores_denominator)

                numenator = torch.sum(torch.exp(scores_numenator))
                denominator = torch.sum(torch.exp(scores_denominator))

                print(numenator, denominator)

                loss += - torch.log(numenator / denominator)
            else:
                loss += self.penality
                
        return loss

    def predict(
        self,
        indices: List[int],
        question_ids: torch.Tensor,
        k: int = 100, 
        verbose: bool = False,
        L: int = 30,
        **kwargs
    ):
        
        N = len(indices)
        
        questions = [self.dataset.questions[id] for id in indices]
        contexts = [self.dataset.contexts[id] for id in indices]
        if verbose:
            print(f"Q: {questions}")
            print(f"C: {contexts}")

        last_hidden_state_start = self.model_start(**question_ids).last_hidden_state.detach().cpu().numpy()[:, 0, :].reshape((N, self.hidden_dim))
        last_hidden_state_end = self.model_end(**question_ids).last_hidden_state.detach().cpu().numpy()[:, 0, :].reshape((N, self.hidden_dim))
        S_start, I_start = self.dump.index.search(np.ascontiguousarray(last_hidden_state_start), k)
        S_end, I_end = self.dump.index.search(np.ascontiguousarray(last_hidden_state_end), k)
        
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
                        
                        context = self.dataset.contexts[self.dump.context_id2id[context_id_candidate_start]]
                        context_ids = self.tokenizer(context)['input_ids']
                        
                        if token_id_start in context_ids and token_id_end in context_ids:
                            start_index = context_ids.index(token_id_start)
                            end_index = context_ids.index(token_id_end)
                            if start_index <= end_index <= start_index + L:
                                answer_candidate_ids = context_ids[start_index:end_index]
                                answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                                answer_candidate2cumscore[(start_index, end_index, answer_candidate)] = (s_start + s_end)  
                        
            if answer_candidate2cumscore:
                answer, score = sorted(answer_candidate2cumscore.items(), key=lambda x: -x[1])[0]
                start_index, end_index, answer = answer
                answers.append(re.sub(r'#', '', answer))
                start_indices.append(start_index)
                end_indices.append(end_index)
                scores.append(score)
            else:
                answers.append('')
                start_indices.append(-1)
                end_indices.append(-1)
                scores.append(-100)
                    
                
        return answers, start_indices, end_indices, scores