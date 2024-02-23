def evaluate(
    gt_answers,
    gt_spans_input_ids_start,
    gt_spans_input_ids_end,
    gt_contexts,
    prediction_answers,
    prediction_start_indices,
    prediction_end_indices,
    k=100
):
    size = len(gt_answers)
    tp_string, tp_start_id, tp_end_id = 0, 0, 0
    delta_start_normalized, delta_end_normalized = 0, 0
    for id in (range(size)):
        answer = prediction_answers[id]
        start_index = prediction_start_indices[id]
        end_index = prediction_end_indices[id]

        if answer == gt_answers[id]:
            tp_string += 1
            
        if start_index == gt_spans_input_ids_start[id]:
            tp_start_id += 1
            
        if end_index == gt_spans_input_ids_end[id]:
            tp_end_id += 1
            
        delta_start_normalized += abs(start_index - gt_spans_input_ids_start[id]) / len(gt_contexts[id])
        delta_end_normalized += abs(end_index - gt_spans_input_ids_end[id]) / len(gt_contexts[id])
        
    return  tp_string / size, tp_start_id / size,  tp_end_id / size, \
        delta_start_normalized  / size, delta_end_normalized / size
    