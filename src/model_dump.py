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
        S_start, I_start = self.index.search(last_hidden_state_start, k)
        S_end, I_end = self.index.search(last_hidden_state_end, k)
        
        
        answers, start_indices, end_indices, scores = [], [], [], []
        for n in range(N):
            answer_candidate2cumscore = {}
            for s_start, token_w_id_start in zip(S_start[n], I_start[n]):
                for s_end, token_w_id_end in zip(S_end[n], I_end[n]):
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
                answers.append(re.sub(r'#', '', answer))
                start_indices.append(start_index)
                end_indices.append(end_index)
                scores.append(score)
        return answers, start_indices, end_indices, scores
        
    

