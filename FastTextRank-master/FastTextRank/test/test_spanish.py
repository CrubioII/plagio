import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FastTextRankSpanish import FastTextRankSpanish, FastTextRankSpanishKeywords

import unittest

class TestFastTextRankSpanish(unittest.TestCase):
    def setUp(self):
        self.texto = (
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )


















    def test_summarize(self):
        extractor = FastTextRankSpanish()
        sentences = extractor.cut_sentences(self.texto)
        original_sentences, filtered_sentences = extractor.tokenize_and_filter(sentences)
        similarity_matrix = extractor.create_similarity_matrix(filtered_sentences)
        scores = extractor.calculate_scores(similarity_matrix)
        resumen = extractor.summarize(self.texto, 2)
        # Obtener los índices y scores de las oraciones seleccionadas
        import heapq
        top_sentences = heapq.nlargest(2, zip(scores, range(len(scores))))
        selected_indices = [idx for _, idx in top_sentences]
        selected_indices.sort()
        print("\nResumen generado:")
        for i, idx in enumerate(selected_indices, 1):
            print(f"{i}. {original_sentences[idx]} (score={scores[idx]:.4f})")
        self.assertIsInstance(resumen, list)
        self.assertTrue(len(resumen) <= 2)
        for oracion in resumen:
            self.assertIsInstance(oracion, str)
            self.assertTrue(len(oracion) > 0)

    def test_keywords(self):
        extractor_kw = FastTextRankSpanishKeywords()
        try:
            words = extractor_kw.extract_keywords(self.texto, 4)
            # Para obtener los scores, necesitamos reconstruir el proceso
            import re
            from collections import OrderedDict
            try:
                tokens = extractor_kw.__class__.__bases__[0].word_tokenize(self.texto, language='spanish')
            except Exception:
                tokens = re.findall(r'\b\w+\b', self.texto.lower())
            if extractor_kw.use_stopword:
                filtered_words = [
                    word.lower() for word in tokens 
                    if word.lower() not in extractor_kw.stop_words and len(word) > 2
                ]
            else:
                filtered_words = [word.lower() for word in tokens if len(word) > 2]
            word_to_index = OrderedDict()
            index_to_word = {}
            for i, word in enumerate(set(filtered_words)):
                word_to_index[word] = i
                index_to_word[i] = word
            num_words = len(word_to_index)
            cooccurrence_matrix = [[0.0] * num_words for _ in range(num_words)]
            for i in range(len(filtered_words)):
                for j in range(max(0, i - extractor_kw.window), min(len(filtered_words), i + extractor_kw.window + 1)):
                    if i != j:
                        word1_idx = word_to_index[filtered_words[i]]
                        word2_idx = word_to_index[filtered_words[j]]
                        cooccurrence_matrix[word1_idx][word2_idx] += 1.0
            scores = extractor_kw.calculate_scores(cooccurrence_matrix)
            import heapq
            top_words = heapq.nlargest(4, zip(scores, range(num_words)))
            print("\nPalabras clave extraídas:")
            for i, (score, idx) in enumerate(top_words, 1):
                print(f"{i}. {index_to_word[idx]} (score={score:.4f})")
        except Exception:
            print("\nPalabras clave extraídas:")
            for i, palabra in enumerate(words, 1):
                print(f"{i}. {palabra}")
        self.assertIsInstance(words, list)
        self.assertTrue(len(words) <= 4)
        for palabra in words:
            self.assertIsInstance(palabra, str)
            self.assertTrue(len(palabra) > 0)

if __name__ == '__main__':
    unittest.main()
