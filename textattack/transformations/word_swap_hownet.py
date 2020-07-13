import pickle

from flair.data import Sentence
from flair.models import SequenceTagger

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapHowNet(WordSwap):
    """Transforms an input by replacing its words with synonyms in the stored
    synonyms bank generated by the OpenHowNet."""

    PATH = "transformations/hownet"

    def __init__(self, max_candidates=-1, **kwargs):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates

        # Download synonym candidates bank if they're not cached.
        cache_path = utils.download_if_needed(
            "{}/{}".format(WordSwapHowNet.PATH, "word_candidates_sense.pkl")
        )

        # Actually load the files from disk.
        with open(cache_path, "rb") as fp:
            self.candidates_bank = pickle.load(fp)

        self._flair_pos_tagger = SequenceTagger.load("pos-fast")
        self.pos_dict = {"JJ": "adj", "NN": "noun", "RB": "adv", "VB": "verb"}

    def _get_replacement_words(self, word, word_pos):
        """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on nearest neighbors selected word embeddings.
        """
        word_pos = self.pos_dict.get(word_pos, None)
        if word_pos is None:
            return []

        try:
            candidate_words = self.candidates_bank[word.lower()][word_pos]
            if self.max_candidates > 0:
                candidate_words = candidate_words[: self.max_candidates]
            return [
                recover_word_case(candidate_word, word)
                for candidate_word in candidate_words
            ]
        except KeyError:
            # This word is not in our synonym bank, so return an empty list.
            return []

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        sentence = Sentence(" ".join(words))
        # in-place POS tagging
        self._flair_pos_tagger.predict(sentence)
        word_list, pos_list = zip_flair_result(sentence)

        assert len(words) == len(
            word_list
        ), "Part-of-speech tagger returned incorrect number of tags"
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            word_to_replace_pos = pos_list[i][:2]  # get the root POS
            replacement_words = self._get_replacement_words(
                word_to_replace, word_to_replace_pos
            )
            transformed_texts_idx = []
            for r in replacement_words:
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["max_candidates"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word


def zip_flair_result(pred):
    """Parse the output from the FLAIR POS tagger."""
    if not isinstance(pred, Sentence):
        raise TypeError("Result from Flair POS tagger must be a `Sentence` object.")

    tokens = pred.tokens
    word_list = []
    pos_list = []
    for token in tokens:
        word_list.append(token.text)
        pos_list.append(token.annotation_layers["pos"][0]._value)

    return word_list, pos_list
