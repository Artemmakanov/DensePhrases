from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import numpy as np



class QueryModel(torch.nn.Module):
    def __init__(self, hidden_dim, H, ds):
        super(QueryModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        model_checkpoint = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.model_1 = AutoModel.from_pretrained(model_checkpoint)
        self.model_2 = AutoModel.from_pretrained(model_checkpoint)
        self.H = torch.Tensor(H)
        self.ds = ds

    def forward(self, id):
        question = self.ds.questions[id]
        context_ids = self.tokenizer(self.ds.contexts[id])['input_ids']
        answer_ids = self.tokenizer(self.ds.answers[id])['input_ids']
        k = 100
        input_ids = self.tokenizer(question, truncation=True, max_length=512, return_tensors="pt")
        last_hidden_state_1 = self.model_1(**input_ids).last_hidden_state[0][0].reshape((1, self.hidden_dim))
        last_hidden_state_2 = self.model_2(**input_ids).last_hidden_state[0][0].reshape((1, self.hidden_dim))
        dot1 = torch.matmul(self.H, last_hidden_state_1.T)
        dot2 = torch.matmul(self.H, last_hidden_state_2.T)
        
        I_1 = np.argsort(-dot1.detach().numpy())[:k]
        I_2 = np.argsort(-dot2.detach().numpy())[:k]
        
        S_1 = dot1[I_1]
        S_2 = dot2[I_2]
        
        answer2cumscore = {}
        answer_correct2cumscore = {}
        for num_i_1, i_1 in enumerate(I_1[0]):
            for num_i_2, i_2 in enumerate(I_2[0]):
                if i_1 in context_ids and i_2 in context_ids:
                    start_index = context_ids.index(i_1)
                    end_index = context_ids.index(i_2)
                    if start_index <= end_index:
                        answer_candidate_ids = context_ids[start_index:end_index]
                        answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                        answer2cumscore[answer_candidate] = torch.Tensor(S_1[0][num_i_1] + S_2[0][num_i_2])
                        if answer_candidate_ids == answer_ids:
                            answer_correct2cumscore[answer_candidate] = torch.Tensor(S_1[0][num_i_1] + S_2[0][num_i_2])
        
        print(answer2cumscore)
        print(answer_correct2cumscore)
        if answer_correct2cumscore:
            scores_numenator = torch.vstack(tuple(answer_correct2cumscore.values()))
        else:
            scores_numenator = torch.Tensor([0.], )
        if answer2cumscore:
            scores_denominator = torch.vstack(tuple(answer2cumscore.values()))
        else:
            scores_denominator = torch.Tensor([0.])
        numenator = torch.sum(torch.exp(scores_numenator))
        denominator = torch.sum(torch.exp(scores_denominator))
        loss = - torch.log(numenator / denominator)
        return loss

