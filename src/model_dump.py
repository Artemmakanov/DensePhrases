from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import faiss




class Dump:
    def __init__(self, ds, model, model_1, model_2, hidden_dim=264):
        self.hidden_dim = hidden_dim
        
        model_checkpoint = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.model = model
        self.model_1 = model_1
        self.model_2 = model_2
        self.ds = ds            

    def create_dump(self):
        token_id2cnt = defaultdict(int)
        token_id2j = {}
        j = 0
        
        for context in tqdm(self.ds.contexts):
            input_ids = self.tokenizer(context, truncation=True, max_length=512, return_tensors="pt")
            last_hidden_state = self.model(**input_ids).last_hidden_state[0].detach().numpy()
            for token_num, token_id in enumerate(input_ids['input_ids'][0]):
                if not token_id in self.tokenizer.added_tokens_decoder.keys(): 
                    last_hidden_state_token = last_hidden_state[token_num].reshape((1, self.hidden_dim))
                    
                    token_id = token_id.item()
                    if token_id in token_id2j:
                        H[token_id2j[token_id]] += last_hidden_state_token[0]
                    else:
                        if j == 0:
                            H = last_hidden_state_token
                        else:
                            print(last_hidden_state_token.shape, H.shape)
                            j += 1
                            H = np.vstack((H, last_hidden_state_token))
                        token_id2j[token_id] = j
                            
                    token_id2cnt[token_id] += 1
                
        for token_id, cnt in token_id2cnt.items():
            H[token_id2j[token_id]] = H[token_id2j[token_id]] / cnt
        self.index = faiss.IndexFlatIP(self.hidden_dim)
        self.index.add(H)
        self.H = H

    def predict(self, id, k=100, verbose=False):
        question = self.ds.questions[id]
        answer = self.ds.answers[id]
        context = self.ds.contexts[id]
        if verbose:
            print(f"Q: {question}")
            print(f"C: {context}")

        input_ids = self.tokenizer(question, truncation=True, max_length=512, return_tensors="pt")
        last_hidden_state_1 = self.model_1(**input_ids).last_hidden_state.detach().numpy()[0][0].reshape((1, self.hidden_dim))
        last_hidden_state_2 = self.model_2(**input_ids).last_hidden_state.detach().numpy()[0][0].reshape((1, self.hidden_dim))
        S_1, I_1 = self.index.search(last_hidden_state_1, k)
        S_2, I_2 = self.index.search(last_hidden_state_2, k)
        
        context_ids =  self.tokenizer(context)['input_ids']
        
        answer_candidate2cumscore = {}
        for num_i_1, i_1 in enumerate(I_1[0]):
            for num_i_2, i_2 in enumerate(I_2[0]):
                if i_1 in context_ids and i_2 in context_ids:
                    start_index = context_ids.index(i_1)
                    end_index = context_ids.index(i_2)
                    if start_index <= end_index:
                        answer_candidate_ids = context_ids[start_index:end_index]
                        answer_candidate = self.tokenizer.decode(answer_candidate_ids)
                        answer_candidate2cumscore[answer_candidate] = S_1[0][num_i_1] + S_2[0][num_i_2]
                        
        if answer_candidate2cumscore:
            answer, score = sorted(answer_candidate2cumscore.items(), key=lambda x: -x[1])[0]
            answer = re.sub(r'#', '', answer)
            return answer, score
        else:
            return '',  0.

    def evaluate(self, k=100):
        tp = 0
        for id in tqdm(range(self.size)):
            answer, score = self.predict(id, k)
            if answer == self.ds.answers[id]:
                tp += 1
        return tp / self.size

