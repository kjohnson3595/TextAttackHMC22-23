'''
Testing T4a's run_attack function
'''

import pytest
from argparse import Namespace

def test_imports():
    # Make sure we can import eukaryote
    from eukaryote.t4a import attack_eval_support, shared
    del attack_eval_support, shared

### Alert: do not run this on the clinic mac! It will take an hour. 
def test_run_attack():
    #Run the simplest possible attack, assert that it works
    from eukaryote.t4a import shared
    from eukaryote.t4a.attack_eval_support import Results, run_attack
    from eukaryote.models.wrappers import HuggingFaceModelWrapper

    # Namespace trick: dicts -> argparse outputs to save headache
    model_arg_dict = {'model_huggingface':'bert-base-uncased'}
    model_wrapper = shared.load_model_wrapper(args=Namespace(**model_arg_dict))
    
    dataset_arg_dict = {'dataset_huggingface':'yelp_polarity', 'dataset_split':'train'}
    dataset = shared.load_dataset(args=Namespace(**dataset_arg_dict))

    res = run_attack(model_wrapper = model_wrapper, dataset = dataset, attack_recipe=shared.attack_recipe_names['alzantot'], perturbation_budget=0.) # first choice of each
    assert isinstance(res, Results) # make sure we return a t4a Results instance
    

