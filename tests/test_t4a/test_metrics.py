'''
Tests for t4a.metrics
'''
import pytest
import numpy as np

from eukaryote.t4a.attack_eval_support import calculate_binary_metrics, calculate_multiclass_metrics

def test_binary_metrics():
    num_examples = 10
    y_true = np.array([1] + [0] * (num_examples-1))
    y_pred_correct = np.array([1] + [0] * (num_examples-1))
    y_scores_correct = np.vstack([np.array([1.,0.]), \
        np.hstack([np.zeros((num_examples-1, 1)), np.ones((num_examples-1, 1))])
        ])
    y_pred_incorrect = np.array([0] + [1] * (num_examples - 1))
    y_scores_incorrect = np.vstack([np.array([0.,1.]),\
        np.hstack([np.ones((num_examples-1, 1)), np.zeros((num_examples-1, 1))])])


    _, correct_metrics = calculate_binary_metrics(y_true=y_true, y_pred=y_pred_correct, y_scores=y_scores_correct)
    _, incorrect_metrics = calculate_binary_metrics(y_true=y_true, y_pred=y_pred_incorrect, y_scores=y_scores_incorrect)

    assert correct_metrics['accuracy'] == 1.
    assert correct_metrics['precision'] == 1.
    
    assert incorrect_metrics['accuracy'] == 0.
    assert incorrect_metrics['precision'] == 0.

def test_multiclass_metrics():
    num_examples = 10
    y_true = np.array([1] + [0] * (num_examples-1))
    y_pred_correct = np.array([1] + [0] * (num_examples-1))
    y_scores_correct = np.vstack([np.array([1.,0.]), \
        np.hstack([np.zeros((num_examples-1, 1)), np.ones((num_examples-1, 1))])
        ])
    y_pred_incorrect = np.array([0] + [1] * (num_examples - 1))
    y_scores_incorrect = np.vstack([np.array([0.,1.]),\
        np.hstack([np.ones((num_examples-1, 1)), np.zeros((num_examples-1, 1))])])


    _, correct_metrics = calculate_multiclass_metrics(y_true=y_true, y_pred=y_pred_correct, y_scores=y_scores_correct)
    _, incorrect_metrics = calculate_multiclass_metrics(y_true=y_true, y_pred=y_pred_incorrect, y_scores=y_scores_incorrect)

    assert correct_metrics['accuracy'] == 1.
    assert correct_metrics['precision'] == 1.
    
    assert incorrect_metrics['accuracy'] == 0.
    assert incorrect_metrics['precision'] == 0.