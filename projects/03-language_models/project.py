# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    # Check robots.txt for the pause length
    robots_url = url[:url.rfind('/')] + '/robots.txt'
    robots_response = requests.get(robots_url)
    pause = 0.5  # Default pause in case robots.txt doesn't specify

    if robots_response.ok:
        robots_txt = robots_response.text
        match = re.search(r'Delay: (\d+)', robots_txt)
        if match:
            pause = int(match.group(1))

    # Pause to adhere to robots.txt policy
    time.sleep(pause)

    # Fetch the content of the book
    corpus = requests.get(url)
    corpus_text = corpus.text

    # Extract the contents without the start and end comments
    corpus_text = corpus_text.split('***')[2:-2]
    corpus_text = ''.join(corpus_text)

    # Replace Windows newline characters with standard newline characters
    corpus_text = corpus_text.replace('\r\n', '\n')

    return corpus_text


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    book_string = re.sub(r'\n{2,}', '\x02', book_string)
    book_string = book_string.replace('\x02', ' \x03 \x02 ')
    tokens = re.findall(r'\b\w+\b|\W', book_string)
    tokens = list(filter(lambda x: x != ' ', tokens))
    tokens = tokens[1:-1]
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):

    def __init__(self, tokens):
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        tokens_index = list(pd.Series(tokens).unique())
        tokens_length = pd.Series(tokens).nunique()
        tokens_values = [1/tokens_length] * tokens_length
        tokens_series = city_series = pd.Series(tokens_values, index=tokens_index)
        return tokens_series
    
    def probability(self, words):
        count = 0
        for word in words:
            if word in (self.mdl.index.to_list()):
                count += 1
        if count==0:
            return count
        else: 
            return (self.mdl.iloc[0])**count
        
    def sample(self, M):
        tokens_sample = np.random.choice(self.mdl.index, replace=True, size=M)
        tokens_sample = ' '.join(tokens_sample)
        return tokens_sample


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):

    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        tokens_series = pd.Series(tokens).value_counts()
        tokens_series = tokens_series / tokens_series.sum()
        return tokens_series
    
    def probability(self, words):
        probs = []
        for word in words:
            if word in (self.mdl.index.to_list()):
                probs += self.mdl.loc[word]
        if len(probs) == 0:
            return 0
        sample_prob = probs[0]
        for probs in probs[1:]: 
            sample_prob *= prob
        return sample_prob
        
    def sample(self, M):
        tokens_sample = np.random.choice(self.mdl.index, replace=True, size=M, p=self.mdl.values)
        tokens_sample = ' '.join(tokens_sample)
        return tokens_sample


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ...
        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        ...
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        ...

        # Create the conditional probabilities
        ...
        
        # Put it all together
        ...
    
    def probability(self, words):
        ...
    

    def sample(self, M):
        # Use a helper function to generate sample tokens of length `length`
        ...
        
        # Transform the tokens to strings
        ...
