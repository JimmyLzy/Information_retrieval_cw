#!/usr/bin/env python3
import pandas as pd 
import os
import numpy as np
import math
import time
import collections
import concurrent.futures
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer

class tf_idf_vectorizer(object):
    def __init__(self):
        self.vocab = []

    def print_vocab(self):
        print(len(self.vocab))


    def fit(self, corpus):
        self.corpus = list(corpus)
        for sentence in corpus:
            tokenized_sentence = self.tokenize(sentence)
            for word in tokenized_sentence:
                self.vocab.append(word)
        self.vocab.append('oov')
        self.vocab = list(set(self.vocab))
        self.vocab = sorted(self.vocab)

    def get_vocab(self):
        return self.vocab

    def tokenize(self, sentence):
        stemmer = PorterStemmer()
        stopwords_english = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
    #     english_words = set(nltk.corpus.words.words()) - stopwords_english
    
        tokenized_words = []
        for word in tokenizer.tokenize(sentence):
            word = stemmer.stem(word.lower())
            if word not in stopwords_english and word.isalpha():
                tokenized_words.append(word)
    
        return tokenized_words

    def transform(self, transform_corpus):
        transform_corpus = list(transform_corpus)
        self.tf_idf_matrix = pd.DataFrame({'sentence': transform_corpus})
        self.tf_idf_matrix.set_index('sentence', inplace=True)

        self.word_idf_dict = collections.defaultdict(float)

        self.calc_word_idf(transform_corpus)

        start_time = time.time()
        print('start calculating tf idf matrix' )

        for word in self.vocab:
            self.tf_idf_matrix[word] = 0.0

        num_text = len(transform_corpus)

        # self.tf_idf_matrix = self.tf_idf_matrix.apply(lambda row : self.calc_tf(row), axis=1)

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start calculate tf-idf for each word
            futures = {executor.submit(self.calc_tf_idf, sentence, num_text): \
                sentence for sentence in transform_corpus}
            for future in concurrent.futures.as_completed(futures):
                future.result()
        # concurrent.futures.wait(futures)
        end_time = time.time()
        print("Time for calc tf idf matrix for all words: %ssecs" % (end_time - start_time))
        return self.tf_idf_matrix


    def calc_word_idf(self, corpus):
        for sentence in corpus:
            sentence = self.tokenize(sentence)
            sentence = self.replace_with_oov(sentence)
            for word in set(sentence):
                self.word_idf_dict[word] += 1.0

    def calc_tf_idf(self, sentence, num_text):

        orgin_sentence = sentence
        sentence = self.tokenize(sentence)
        sentence = self.replace_with_oov(sentence)

        for word in set(sentence):
            tf = sentence.count(word)
            idf = self.word_idf_dict[word]
            idf =  math.log(num_text / idf) if idf else 0
            self.tf_idf_matrix.loc[self.tf_idf_matrix.index==orgin_sentence, word] = tf * idf

        print(str(sentence[:1]) + ' is completed!\n')          
    
    def replace_with_oov(self, sentence):
        res = []
        for word in sentence:
            if word in self.vocab:
                res.append(word)
            else:
                res.append('oov')
        return res