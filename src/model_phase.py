from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer, AutoModel
from typing import List
import torch

from tqdm import tqdm

class PhraseModel(torch.nn.Module):
    def __init__(self, hidden_dim, dataset, model_checkpoint, device):
        super(PhraseModel, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.model_start = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.model_end = AutoModel.from_pretrained(model_checkpoint).to(self.device)
        self.dataset = dataset
        self.softmax = torch.nn.Softmax()
        

    def forward(
        self,
        context_ids: torch.Tensor,
        question_ids: torch.Tensor,
        start_token_indices: List[int],
        end_token_indices: List[int],
        **kwargs
    ):

        last_hidden_state = self.model(**context_ids).last_hidden_state[:, 1:-1, :]
        
        N, context_num_tokens, _ = last_hidden_state.shape
        q_start = self.model_start(**question_ids).last_hidden_state[:, 0, :]
        q_end = self.model_end(**question_ids).last_hidden_state[:, 0, :]
  
        loss = torch.tensor([0.], requires_grad=True).to(self.device)

        for n in range(N):
            
            z_start = torch.matmul(last_hidden_state[n], q_start[n].T).reshape(context_num_tokens)
            P_start = self.softmax(z_start)
            loss_start = torch.log(P_start[start_token_indices[n]])
            
            z_end = torch.matmul(last_hidden_state[n], q_end[n].T).reshape(context_num_tokens)
            P_end = self.softmax(z_end)
            loss_end = torch.log(P_end[end_token_indices[n]])
            
            loss = loss + (loss_start + loss_end) / 2
            
        return -loss