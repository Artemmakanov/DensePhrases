import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Dataset_QA(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        contexts,
        questions,
        answers,
        size,
        device='cpu'
    ):
        self.tokenizer = tokenizer
        self.ids = []
        self.contexts = []
        self.questions = []
        self.answers = []
        self.spans_input_ids = []
        id = 0
        
        if size:
            treshold = size
            self.size = size
        else:
            treshold = len(contexts)
        for i in tqdm(range(treshold)):
            context = contexts[i]
            answer = answers[i]

            answer_tokenized = self.tokenizer(answer)['input_ids'][1:-1]
            context_tokenized = self.tokenizer(context)['input_ids'][1:-1]
                    
            for context_start_token in range(len(context_tokenized) - len(answer_tokenized)):
                
                if answer_tokenized == context_tokenized[context_start_token:context_start_token+len(answer_tokenized)]:
                    break
                context_start_token += 1

            
            context_end_token = context_start_token + len(answer_tokenized)

            if context_start_token < 512 and context_end_token < 512: 
                self.spans_input_ids.append({
                    'start':context_start_token,
                    'end': context_end_token
                })
                
                self.contexts.append(contexts[i])
                self.questions.append(questions[i])
                self.answers.append(answer)

                self.ids.append(id)
                id += 1
           
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    def __getitem__(self, idx):
        inputs_context = squeeze_dict(self.tokenizer(
            self.contexts[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )).to(self.device)

        inputs_question = squeeze_dict(self.tokenizer(
            self.questions[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )).to(self.device)

        return {'context_ids': inputs_context,
                'question_ids': inputs_question,
                'start_token_indices': self.spans_input_ids[idx]['start'],
                'end_token_indices': self.spans_input_ids[idx]['end']
               }
        
    def as_data_loader(self, batch_size=10):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
    
    def __len__(self):
        return len(self.contexts)
    
def squeeze_dict(dct):
    for k, w in dct.items():
        dct[k] = w[0]
    return dct
