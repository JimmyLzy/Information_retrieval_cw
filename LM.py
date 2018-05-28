#!/usr/bin/env python3
import pandas as pd 
import os
import numpy as np
import math
import time
import collections
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer

class Unigram(object):

    def __init__(self, corpus, epsilon):
        self.prob_dict = collections.defaultdict(float)
        self.order = 1

        corpus_length = len(corpus)

        self.vocab = list(set(corpus))
        vocab_length = len(self.vocab)

        for word in self.vocab:
            self.prob_dict[word] = (corpus.count(word) + epsilon) / \
                (corpus_length + epsilon * vocab_length)


    def probability(self, word):
        return self.prob_dict[word] if word in self.vocab else self.prob_dict['oov']


class Ngram(object):

    def __init__(self, corpus, order, epsilon):
        self.counts_dict = collections.defaultdict(float)
        self.norm_dict = collections.defaultdict(float)
        self.order = order
        self.vocab = list(set(corpus))
        self.epsilon = epsilon

        for i in range(self.order - 1, len(corpus)):
            history = tuple(corpus[i-self.order+1:i])
            word = corpus[i]
            self.counts_dict[(word,) + history] += 1
            self.norm_dict[history] += 1

    def probability(self, word, *history):
        sub_history = tuple(history[-(self.order-1):]) if self.order > 1 else ()
        return (self.counts_dict[(word,) + sub_history] + self.epsilon) \
            / (self.norm_dict[sub_history] + self.epsilon * len(self.vocab))