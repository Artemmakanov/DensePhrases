from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from transformers import AutoModel
from typing import List
import torch

from tqdm import tqdm

class PhraseModel(torch.nn.Module):
    def __init__(self, hidden_dim, ds):
        super(PhraseModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        model_checkpoint = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.model_1 = AutoModel.from_pretrained(model_checkpoint)
        self.model_2 = AutoModel.from_pretrained(model_checkpoint)
        self.ds = ds
        self.softmax = torch.nn.Softmax()
        

    def forward(self, ids: List[int]):
        context_ids = self.tokenizer(self.ds.get_contexts(ids), 
                                     truncation=True, max_length=512, 
                                     return_tensors="pt", padding=True)
        question_ids = self.tokenizer(self.ds.get_questions(ids), 
                                      truncation=True, max_length=512, 
                                      return_tensors="pt", padding=True)
        last_hidden_state = self.model(**context_ids).last_hidden_state

        N, context_num_tokens, _ = last_hidden_state.shape
        q_start = self.model_1(**question_ids).last_hidden_state[:, 1]
        q_end = self.model_2(**question_ids).last_hidden_state[:, -1]
        start_token_indices = self.ds.get_spans_input_ids(ids, pos='start')
        end_token_indices = self.ds.get_spans_input_ids(ids, pos='end')
        loss = torch.Tensor([0.])
        for n in range(N):       
            z_start = torch.matmul(last_hidden_state[n], q_start[n].T).reshape(context_num_tokens)
            P_start = self.softmax(z_start)
            loss_start = P_start[start_token_indices[n]]
            
            z_end = torch.matmul(last_hidden_state[n], q_end[n].T).reshape(context_num_tokens)
            P_end = self.softmax(z_end)
            loss_end = P_end[end_token_indices[n]]
            
            loss += (loss_start + loss_end)

        return -loss