import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_QA(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.context = df.context.values
        self.question = df.question.values
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
                'question_ids': inputs_question
               }

    def __len__(self):
        return len(self.context)
