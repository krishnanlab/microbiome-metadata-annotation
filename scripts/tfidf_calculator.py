from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from collections import Counter


class TfidfCalculator:
    """tf idf tfidf calculator for single words in corpus"""

    def __init__(self, data):
        self.data = data
        self.idf = None
        self.tf = None
        self.tfidf = None

    def calculate_tfidf(self):
        """Calculate TF-IDF values for single words in corpus"""
        vectorizer = TfidfVectorizer(norm=None)
        self.tfidf = vectorizer.fit_transform(self.data).toarray()
        return self.tfidf

    def calculate_tf(self):
        """Calculate TF values (count) for single words in corpus"""
        vectorizer = CountVectorizer()
        self.tf = vectorizer.fit_transform(self.data).toarray()
        return self.tf

    def calculate_idf(self):
        """Calculate IDF values for single words in corpus"""
        vectorizer = TfidfVectorizer()
        vectorizer.fit(self.data)
        self.idf = vectorizer.idf_
        return self.idf

    def get_word_features(self):
        """get word features of returning tf/idf/tfidf matrix"""
        vectorizer = TfidfVectorizer()
        vectorizer.fit(self.data)
        return vectorizer.get_feature_names_out()
