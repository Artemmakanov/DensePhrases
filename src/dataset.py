from pathlib import Path
parent_dir = Path(__file__).parent.parent


from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List

class Dataset:
    
    def __init__(self) -> None:
        
        self.ids = []
        self.contexts = []
        self.questions = []
        self.answers = []
        self.spans_input_ids = []
        model_checkpoint = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
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
            try:
                answer_tokenized = self.tokenizer(answers[i]['text'][0])['input_ids'][1:]
                context_tokenized = self.tokenizer(contexts[i])['input_ids'][1:]
                start = context_tokenized.index(answer_tokenized[0])
                end = context_tokenized.index(answer_tokenized[-1])
                if start < 512 and end < 512:
                    self.spans_input_ids.append({
                        'start':start,
                        'end': end
                    })
                    
                    self.contexts.append(contexts[i])
                    self.questions.append(questions[i])
                    self.answers.append(answers[i]['text'])
                    
                    self.ids.append(id)
                    id += 1
            except:
                None