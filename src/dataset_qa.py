import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Dataset_QA(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        contexts,
        questions,
        answers,
        size
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
    
    def __getitem__(self, idx):
        inputs_context = self.tokenizer.encode(
            self.context[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        inputs_question = self.tokenizer.encode(
            self.question[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        return {'context_ids': inputs_context,
                'question_ids': inputs_question,
                'span_input_start_ids': self.spans_input_ids[idx]['start'],
                'span_input_end_ids': self.spans_input_ids[idx]['ends']
               }
        
    def __len__(self):
        return len(self.context)
