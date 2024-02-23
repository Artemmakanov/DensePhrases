from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import faiss

import re



class Dump:
    def __init__(self, ds, model, model_1, model_2, tokenizer, hidden_dim=264, device='cuda'):
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer
        self.model = model
        self.model_1 = model_1
        self.model_2 = model_2
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

    def predict(self, id, k=100, verbose=False, L=30):
        question = self.ds.questions[id]
        answer = self.ds.answers[id]
        context = self.ds.contexts[id]
        if verbose:
            print(f"Q: {question}")
            print(f"C: {context}")

        input_ids = self.tokenizer(question, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        last_hidden_state_start = self.model_1(**input_ids).last_hidden_state.detach().cpu().numpy()[0][0].reshape((1, self.hidden_dim))
        last_hidden_state_end = self.model_2(**input_ids).last_hidden_state.detach().cpu().numpy()[0][0].reshape((1, self.hidden_dim))
        S_start, I_start = self.index.search(last_hidden_state_start, k)
        S_end, I_end = self.index.search(last_hidden_state_end, k)
        
        
        
        answer_candidate2cumscore = {}
        for s_start, token_w_id_start in zip(S_start[0], I_start[0]):
            for s_end, token_w_id_end in zip(S_end[0], I_end[0]):
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
                        if start_index <= end_index <= start_index + L:
                            answer_candidate_ids = context_ids[start_index:end_index]
                            answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                            answer_candidate2cumscore[(start_index, end_index, answer_candidate)] = s_start + s_end
                            
                    
                        
        if answer_candidate2cumscore:
            answer, score = sorted(answer_candidate2cumscore.items(), key=lambda x: -x[1])[0]
            start_index, end_index, answer = answer
            answer = re.sub(r'#', '', answer)
            return answer, start_index, end_index, score
        else:
            return '',  0.

    def evaluate(self, k=100):
        tp_string, tp_start_id, tp_end_id = 0, 0, 0
        for id in tqdm(range(self.size)):
            answer, start_index, end_index, score = self.predict(id, k)
            if answer == self.ds.answers[id]:
                tp_string += 1
                
            if start_index == self.ds.spans_input_ids[id]['start']:
                tp_start_id += 1
                
            if end_index == self.ds.spans_input_ids[id]['end']:
                tp_end_id += 1
        return {
            "em_string": tp_string / self.size, 
            "em_start": tp_start_id / self.size,
            "em_end": tp_end_id / self.size,
        }

