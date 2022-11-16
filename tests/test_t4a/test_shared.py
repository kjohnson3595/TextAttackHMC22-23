'''
Tests for t4a.shared module
'''
from argparse import Namespace
from eukaryote.t4a import shared


def test_load_model_huggingface():
    from eukaryote.models.wrappers import HuggingFaceModelWrapper
    model_arg_dict = {'model_huggingface':'distilbert-base-uncased'}
    model = shared.load_model_wrapper(args=Namespace(**model_arg_dict))
    assert isinstance(model, HuggingFaceModelWrapper)
    del HuggingFaceModelWrapper

def test_load_dataset_huggingface():
    from eukaryote.datasets import HuggingFaceDataset
    dataset_arg_dict = {'dataset_huggingface':'yelp_polarity', 'dataset_split':'train'}
    dataset = shared.load_dataset(args=Namespace(**dataset_arg_dict))

    assert isinstance(dataset, HuggingFaceDataset)
    del HuggingFaceDataset