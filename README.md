<h1 align="center">T4A HMC2022 🐙</h1>

<p align="center">A lightweight framework for testing adversarial examples on NLP models</p>

# For developers

## Installation
Create a new conda environment with python 3.8:

```$ conda create -n <env_name> python=3.8```

```$ conda activate <env_name>```

```(<env_name>) $ pip install git+https://github.com/kjohnson3595/TextAttackHMC22-23.git@development```

## Functionality

All of TextAttack's original functionality is still included, see [their documentation](https://textattack.readthedocs.io/en/latest/) for instructions on how to use it.

To use the interactive CLI for a walkthrough of the core functionality T4A adds to TextAttack:

```python -m euclids interactive```

T4A's full functionality can be accessed through three commands:
- ```python -m euclids t4a_attack_eval```, which evaluates a model's performance against an attack
- ```python -m euclids t4a_attack_train```, which trains a model on attacked text
- ```python -m euclids t4a_train```, which trains a model

More detailed documentation for each of these commands can be accessed with the ```-h``` flag.


# Contributing

## Installation
1. Clone the source repository: 

```$ git clone git@github.com:kjohnson3595/TextAttackHMC22-23.git```

2. Install all runtime and development dependencies in a new conda env:

```$ conda create -f env.yml```

```$ conda activate t4a```

3. Install the eukaryote package:

```$ pip install -e .```

## Running tests
Tests will run by default every time you push code to a new branch (for now). Before pushing, it's generally a good idea to run the tests on your own machine.
If you aren't working on code coverage and just want to run tests, run the following command:

```$ pytest tests -v```

If you want to assess test coverage, run the following commands:

```$ coverage run -m pytest tests```

```$ coverage report -m``` (this will print a nice report of per-file and overall code coverage)




## Citing TextAttack

This code is a fork and an extension of the original TextAttack library with additional features for Proofpoint's model testing purposes.

If you use code from this repo for your research, please cite [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909).

```bibtex
@inproceedings{morris2020textattack,
  title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
  author={Morris, John and Lifland, Eli and Yoo, Jin Yong and Grigsby, Jake and Jin, Di and Qi, Yanjun},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={119--126},
  year={2020}
}
```


