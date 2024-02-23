from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List
import re

class Dataset:
    
    def __init__(self, model_checkpoint) -> None:
        
        self.ids = []
        self.contexts = []
        self.questions = []
        self.answers = []
        self.spans_input_ids = []
        self.context_embeddings = []
        self.answer_embeddings = []
        self.question_embeddings = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def get_contexts(self, indices: List[int]):
        return [self.contexts[id] for id in indices]
    
    def get_questions(self, indices: List[int]):
        return [self.questions[id] for id in indices]
    
    def get_spans_input_ids(self, indices: List[int], pos: str):
        return [self.spans_input_ids[id][pos] for id in indices]
    
    
    def load_ds(self, raw_datasets, size=100):

        id = 0
        contexts = raw_datasets['train']['context'] + raw_datasets['validation']['context']
        questions = raw_datasets['train']['question'] + raw_datasets['validation']['question']
        answers = raw_datasets['train']['answers'] + raw_datasets['validation']['answers']
        if size:
            treshold = size
            self.size = size
        else:
            treshold = len(contexts)
        for i in tqdm(range(treshold)):
            context = contexts[i]
            answer = answers[i]["text"][0]

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
           