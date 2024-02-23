from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer, AutoModel
from typing import List
import torch

from tqdm import tqdm

class PhraseModel(torch.nn.Module):
    def __init__(self, hidden_dim, ds, model_checkpoint, device):
        super(PhraseModel, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.model_start = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.model_end = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.ds = ds
        self.softmax = torch.nn.Softmax()
        

    def forward(self, ids: List[int]):
        context_ids = self.tokenizer(self.ds.get_contexts(ids), 
                                     truncation=True, max_length=512, 
                                     return_tensors="pt", padding=True).to(self.device)
        question_ids = self.tokenizer(self.ds.get_questions(ids), 
                                      truncation=True, max_length=512, 
                                      return_tensors="pt", padding=True).to(self.device)
        last_hidden_state = self.model(**context_ids).last_hidden_state[1:-1]
        
        N, context_num_tokens, _ = last_hidden_state.shape
        q_start = self.model_start(**question_ids).last_hidden_state[:, 0]
        q_end = self.model_end(**question_ids).last_hidden_state[:, 0]
        start_token_indices = self.ds.get_spans_input_ids(ids, pos='start')
        end_token_indices = self.ds.get_spans_input_ids(ids, pos='end')
        loss = torch.Tensor([0.]).to(self.device)
        # print(f"context_num_tokens {context_num_tokens}")
        # print(f"last_hidden_state.shape {last_hidden_state.shape}")
        # print(f"q_start.shape {q_start.shape}")
        # print(f"q_end.shape {q_end.shape}")
        # print(f"start_token_indices {start_token_indices}")
        # print(f"end_token_indices {end_token_indices}")
        for n in range(N):
            
            z_start = torch.matmul(last_hidden_state[n], q_start[n].T).reshape(context_num_tokens)
            P_start = self.softmax(z_start)
            loss_start = torch.log(P_start[start_token_indices[n]])
            
            z_end = torch.matmul(last_hidden_state[n], q_end[n].T).reshape(context_num_tokens)
            P_end = self.softmax(z_end)
            loss_end = torch.log(P_end[end_token_indices[n]])
            
            loss += (loss_start + loss_end) / 2

        return -loss