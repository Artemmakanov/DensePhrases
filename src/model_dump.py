from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import faiss

import re



class Dump:
    def __init__(self, ds, model, model_start, model_end, tokenizer, hidden_dim=768, device='cuda'):
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer
        self.model = model
        self.model_start = model_start
        self.model_end = model_end
        self.ds = ds            
        self.device = device

    def create_dump(self):
        self.token_w_id2context_id = {}
        self.token_w_id2token_id = {}
        self.context_id2id = {}
        token_w_id = 0
        
        contexts_unique = list(set(self.ds.contexts))
        for context_id, context_unique in enumerate(contexts_unique):
            self.context_id2id[context_id] = self.ds.contexts.index(context_unique)
        
        
        for context_id, context in enumerate(tqdm(contexts_unique, desc='Creating Phrase dump')):
            input_ids = self.tokenizer(context, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            last_hidden_state = self.model(**input_ids).last_hidden_state[0].detach().cpu().numpy()
            for token_num, token_id in enumerate(input_ids['input_ids'][0]):
                token_id = token_id.item()
                if not token_id in self.tokenizer.added_tokens_decoder.keys(): 
                    last_hidden_state_token = last_hidden_state[token_num].reshape((1, self.hidden_dim))
                    
                    if token_w_id == 0:
                        H = last_hidden_state_token
                    else:
                        H = np.vstack((H, last_hidden_state_token))
                        
                    self.token_w_id2context_id[token_w_id] = context_id
                    self.token_w_id2token_id[token_w_id] = token_id
                    token_w_id += 1 
                
        self.index = faiss.IndexFlatIP(self.hidden_dim)
        self.index.add(H)
        self.H = H
        size, hiiden = H.shape
        self.size = len(self.ds.contexts)

    
        
    

