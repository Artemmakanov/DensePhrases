from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import numpy as np



class QueryModel(torch.nn.Module):
    def __init__(
        self,
        model,
        model_start,
        model_end,
        hidden_dim,
        H,
        ds,
        L=30,
        device='cpu'
    ):
        super(QueryModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = model
        self.model_start = model_start
        self.model_end = model_end
        self.H = torch.Tensor(H)
        self.ds = ds
        self.L = L
        self.device = device

    def forward(self, id):
        question = self.ds.questions[id]
        context_ids = self.tokenizer(self.ds.contexts[id])['input_ids'].to(device)
        answer_ids = self.tokenizer(self.ds.answers[id])['input_ids'].to(device)
        k = 100
        input_ids = self.tokenizer(question, truncation=True, max_length=512, return_tensors="pt").to(device)
        last_hidden_state_1 = self.model_1(**input_ids).last_hidden_state[0][0].reshape((1, self.hidden_dim))
        last_hidden_state_2 = self.model_2(**input_ids).last_hidden_state[0][0].reshape((1, self.hidden_dim))
        dot1 = torch.matmul(self.H, last_hidden_state_1.T)
        dot2 = torch.matmul(self.H, last_hidden_state_2.T)
        
        I_start = np.argsort(-dot1.detach().numpy())[:k]
        I_end = np.argsort(-dot2.detach().numpy())[:k]
        
        S_start = dot1[I_start]
        S_end = dot2[I_end]
        
        answer_candidate2cumscore = {}
        for s_start, token_w_id_start in zip(S_start, I_start):
            for s_end, token_w_id_end in zip(S_end, I_end):
                context_id_candidate_start = self.token_w_id2context_id[token_w_id_start]
                context_id_candidate_end = self.token_w_id2context_id[token_w_id_end]
                if context_id_candidate_start == context_id_candidate_end:
                    
                    token_id_start = self.token_w_id2token_id[token_w_id_start]
                    token_id_end = self.token_w_id2token_id[token_w_id_end]
                    
                    context = self.ds.contexts[self.context_id2id[context_id_candidate_start]]
                    context_ids = self.tokenizer(context)['input_ids']
                    
                    if token_id_start in context_ids and token_id_end in context_ids:
                        start_index = context_ids.index(token_id_start)
                        end_index = context_ids.index(token_id_end)
                        if start_index <= end_index <= start_index + self.L:
                            answer_candidate_ids = context_ids[start_index:end_index]
                            answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                            answer_candidate2cumscore[(start_index, end_index, answer_candidate)] = s_start + s_end
                            
        
        print(answer_candidate2cumscore)
        if answer_candidate2cumscore:
            scores_numenator = torch.vstack(tuple(answer_candidate2cumscore.values()))
        else:
            scores_numenator = torch.Tensor([0.], )
        if answer_candidate2cumscore:
            scores_denominator = torch.vstack(tuple(answer_candidate2cumscore.values()))
        else:
            scores_denominator = torch.Tensor([0.])
        numenator = torch.sum(torch.exp(scores_numenator))
        denominator = torch.sum(torch.exp(scores_denominator))
        loss = - torch.log(numenator / denominator)
        return loss

