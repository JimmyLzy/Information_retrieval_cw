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
        self.tokenized_corpus = []
        self.corpus = list(corpus)
        for sentence in corpus:
            tokenized_sentence = self.tokenize(sentence)
            self.tokenized_corpus.append(tokenized_sentence)
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
        self.tf_idf_martix = pd.DataFrame({'sentence': transform_corpus})
        self.tf_idf_martix.set_index('sentence', inplace=True)

        start_time = time.time()
        print('start calculating tf idf matrix' )

        for word in self.vocab:
            self.tf_idf_martix[word] = 0.0

        num_text = len(transform_corpus)
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start calculate tf-idf for each word
            futures = {executor.submit(self.calc_tfidf, word, num_text): \
                word for word in self.vocab}
            for future in concurrent.futures.as_completed(futures):
                future.result()
        # concurrent.futures.wait(futures)
        end_time = time.time()
        print("Time for calc_tfidf words: %ssecs" % (end_time - start_time))
        return self.tf_idf_martix

    def calc_tfidf(self, word, num_text):

        idf = 0.0
        # if word not in self.vocab:
        #     word = 'oov'

        for index, row in self.tf_idf_martix.iterrows():
            sentence = self.tokenize(index)
            sentence = self.replace_with_oov(sentence)
            if word in sentence:
                tf = sentence.count(word) / len(sentence)
                row[word] = tf
                idf += 1.0

        idf = math.log(num_text / idf, 10) if idf else 0

        self.tf_idf_martix[word] *= idf


        print(word + ' is completed!\n')


    def replace_with_oov(self, sentence):
        res = []
        for word in sentence:
            if word in self.vocab:
                res.append(word)
            else:
                res.append('oov')
        return res
