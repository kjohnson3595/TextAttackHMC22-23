## Attack Success rate calculation function spec
from typing import List
import torch

def calc_asr_from_preds(original_preds: torch.Tensor, 
                        attacked_preds: torch.Tensor,
                        num_examples_per_source_sentence: List[int]):
    '''
    Given two sets of predictions on the same dataset from the same model
        where attacked_preds represents predictions on the dataset after some AA,
        and num_examples_per_source_sentence[i] is the number of sentences code-mixed from 
        sentence i in the original predictions, calculate the ASR. 
    '''
    expanded_preds = []
    for label in original_preds:
        expanded_preds.append(label * torch.ones(num_examples_per_source_sentence))
    expanded_original_preds = torch.cat(expanded_preds)

    unweighted_asr = (expanded_original_preds - attacked_preds)/len(attacked_preds)
    
    return unweighted_asr