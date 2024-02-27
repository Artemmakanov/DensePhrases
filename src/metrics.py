from typing import List

def evaluate(
    answers: List[str],
    start_token_indices: List[str],
    end_token_indices: List[str],
    contexts: List[str],
    prediction_answers: List[str],
    prediction_start_indices: List[int],
    prediction_end_indices: List[int],
    k: int=100,
    **kwargs
):
    size = len(answers)
    tp_string, tp_start_id, tp_end_id = 0, 0, 0
    delta_start_normalized, delta_end_normalized = 0, 0
    for id in (range(size)):
        answer = prediction_answers[id]
        start_index = prediction_start_indices[id]
        end_index = prediction_end_indices[id]

        if answer == answers[id]:
            tp_string += 1
            
        if start_index == start_token_indices[id]:
            tp_start_id += 1
            
        if end_index == end_token_indices[id]:
            tp_end_id += 1
            
        delta_start_normalized += abs(start_index - start_token_indices[id]) / len(contexts[id])
        delta_end_normalized += abs(end_index - end_token_indices[id]) / len(contexts[id])
        
    return  tp_string / size, tp_start_id / size,  tp_end_id / size, \
        delta_start_normalized  / size, delta_end_normalized / size
    