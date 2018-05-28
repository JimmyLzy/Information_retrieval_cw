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

class interpolated_lm(object):

    def __init__(self, lm, backoff, alpha):
        self.lm = lm
        self.backoff = backoff
        self.alpha = alpha
        self.order = lm.order
        self.vocab = lm.vocab


    def probability(self, word, *history):
        return self.alpha * self.lm.probability(word, *history) + (1 - self.alpha) \
            * self.backoff.probability(word, *history)


