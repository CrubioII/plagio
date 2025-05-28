import nltk
import math
import re
from string import punctuation
from heapq import nlargest
from itertools import product, count
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class FastTextRankSpanish:
    def __init__(self, use_stopword=True, custom_stopwords=None, max_iter=100, tol=0.0001):
        self.use_stopword = use_stopword
        self.max_iter = max_iter
        self.tol = tol
        if use_stopword:
            if custom_stopwords:
                self.stop_words = set(custom_stopwords)
            else:
                try:
                    spanish_stopwords = set(stopwords.words('spanish'))
                    self.stop_words = spanish_stopwords.union(set(punctuation))
                    self.stop_words.update(['...', '–', '—', '"', '"', ''', '''])
                except:
                    self.stop_words = {
                        'a', 'ante', 'bajo', 'con', 'de', 'desde', 'en', 'entre', 'hacia', 
                        'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras', 'el', 
                        'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero',
                        'si', 'no', 'que', 'como', 'cuando', 'donde', 'quien', 'cual',
                        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
                        'aquel', 'aquella', 'aquellos', 'aquellas', 'yo', 'tu', 'el',
                        'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te',
                        'se', 'nos', 'os', 'le', 'les', 'lo', 'la', 'los', 'las'
                    }
        else:
            self.stop_words = set()

    def cut_sentences(self, text):
        try:
            sentences = sent_tokenize(text, language='spanish')
        except:
            sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_and_filter(self, sentences):
        original_sentences = []
        filtered_sentences = []
        for sentence in sentences:
            original_sentences.append(sentence)
            try:
                words = word_tokenize(sentence, language='spanish')
            except:
                words = re.findall(r'\b\w+\b', sentence.lower())
            if self.use_stopword:
                filtered_words = [
                    word.lower() for word in words 
                    if word.lower() not in self.stop_words and len(word) > 2
                ]
            else:
                filtered_words = [word.lower() for word in words if len(word) > 2]
            filtered_sentences.append(filtered_words)
        return original_sentences, filtered_sentences

    def calculate_similarity(self, sent1, sent2):
        if not sent1 or not sent2:
            return 0.0
        common_words = 0
        for word in sent1:
            if word in sent2:
                common_words += 1
        if common_words == 0:
            return 0.0
        return common_words / math.log(len(sent1) + len(sent2))

    def create_similarity_matrix(self, sentences):
        num_sentences = len(sentences)
        similarity_matrix = [[0.0 for _ in range(num_sentences)] for _ in range(num_sentences)]
        for i, j in product(range(num_sentences), repeat=2):
            if i != j:
                similarity_matrix[i][j] = self.calculate_similarity(sentences[i], sentences[j])
        return similarity_matrix

    def calculate_scores(self, similarity_matrix):
        n = len(similarity_matrix)
        scores = [0.5] * n
        old_scores = [0.0] * n
        denominators = []
        for i in range(n):
            total = sum(similarity_matrix[i])
            denominators.append(total if total > 0 else 1.0)
        for iteration in range(self.max_iter):
            old_scores = scores[:]
            for i in range(n):
                rank = 0.0
                for j in range(n):
                    if i != j:
                        rank += similarity_matrix[j][i] / denominators[j]
                scores[i] = 0.15 + 0.85 * rank
            if self.has_converged(scores, old_scores):
                break
        return scores

    def has_converged(self, scores, old_scores):
        for i in range(len(scores)):
            if abs(scores[i] - old_scores[i]) >= self.tol:
                return False
        return True

    def summarize(self, text, num_sentences):
        sentences = self.cut_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences
        original_sentences, filtered_sentences = self.tokenize_and_filter(sentences)
        similarity_matrix = self.create_similarity_matrix(filtered_sentences)
        scores = self.calculate_scores(similarity_matrix)
        top_sentences = nlargest(num_sentences, zip(scores, count()))
        selected_indices = [idx for _, idx in top_sentences]
        selected_indices.sort()
        return [original_sentences[i] for i in selected_indices]

class FastTextRankSpanishKeywords:
    def __init__(self, use_stopword=True, custom_stopwords=None, max_iter=100, tol=0.0001, window=3):
        self.use_stopword = use_stopword
        self.max_iter = max_iter
        self.tol = tol
        self.window = window
        if use_stopword:
            if custom_stopwords:
                self.stop_words = set(custom_stopwords)
            else:
                try:
                    spanish_stopwords = set(stopwords.words('spanish'))
                    self.stop_words = spanish_stopwords.union(set(punctuation))
                except:
                    self.stop_words = {
                        'a', 'ante', 'bajo', 'con', 'de', 'desde', 'en', 'entre', 'hacia', 
                        'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras', 'el', 
                        'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero',
                        'si', 'no', 'que', 'como', 'cuando', 'donde', 'quien', 'cual'
                    }
        else:
            self.stop_words = set()

    def extract_keywords(self, text, num_keywords):
        try:
            words = word_tokenize(text, language='spanish')
        except:
            words = re.findall(r'\b\w+\b', text.lower())
        if self.use_stopword:
            filtered_words = [
                word.lower() for word in words 
                if word.lower() not in self.stop_words and len(word) > 2
            ]
        else:
            filtered_words = [word.lower() for word in words if len(word) > 2]
        if not filtered_words:
            return []
        word_to_index = {}
        index_to_word = {}
        for i, word in enumerate(set(filtered_words)):
            word_to_index[word] = i
            index_to_word[i] = word
        num_words = len(word_to_index)
        cooccurrence_matrix = [[0.0] * num_words for _ in range(num_words)]
        for i in range(len(filtered_words)):
            for j in range(max(0, i - self.window), min(len(filtered_words), i + self.window + 1)):
                if i != j:
                    word1_idx = word_to_index[filtered_words[i]]
                    word2_idx = word_to_index[filtered_words[j]]
                    cooccurrence_matrix[word1_idx][word2_idx] += 1.0
        scores = self.calculate_scores(cooccurrence_matrix)
        top_words = nlargest(num_keywords, zip(scores, range(num_words)))
        return [index_to_word[idx] for _, idx in top_words]

    def calculate_scores(self, matrix):
        n = len(matrix)
        scores = [0.5] * n
        old_scores = [0.0] * n
        denominators = []
        for i in range(n):
            total = sum(matrix[i])
            denominators.append(total if total > 0 else 1.0)
        for iteration in range(self.max_iter):
            old_scores = scores[:]
            for i in range(n):
                rank = 0.0
                for j in range(n):
                    if i != j:
                        rank += matrix[j][i] / denominators[j]
                scores[i] = 0.15 + 0.85 * rank
            converged = True
            for i in range(n):
                if abs(scores[i] - old_scores[i]) >= self.tol:
                    converged = False
                    break
            if converged:
                break
        return scores
