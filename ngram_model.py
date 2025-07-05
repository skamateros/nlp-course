from collections.abc import Iterable
import math

class NGramModel:
    """
    NGramModel implements an n-gram language model with add-k smoothing.

    Args:
        n (int): The order of the n-gram model (e.g., 2 for bigram, 3 for trigram).
        k (float): Smoothing parameter for add-k smoothing.
        sentences (Iterable[tuple[str]]): An iterable of sentences, where each sentence is a tuple of words.

    Attributes:
        n (int): The order of the n-gram model.
        k (float): Smoothing parameter.
        n_grams (dict): Dictionary mapping n-gram tuples to their counts.
        n_grams_minus1 (dict): Dictionary mapping (n-1)-gram tuples to their counts.
        uniquewords (set): Set of unique words in the training data (including 'PAD' and 'END').
        V (int): Vocabulary size (number of unique words).
        kV (float): Product of k and V, used for smoothing.

    Methods:
        p(word: str, context: tuple[str]) -> float:
            Returns the log-probability of `word` following the given `context` (of length n-1),
            using add-k smoothing.
        score(sentence: tuple[str]) -> float:
            Returns the total log-probability of the given sentence under the model,
            computed as the sum of the log-probabilities of each word given its context.
    """

    def __init__(self, n: int, k: float, sentences: Iterable[tuple[str]]):
        self.n = n
        self.k = k
        self.n_grams = {}
        self.n_grams_minus1 = {}
        self.uniquewords = set()
        
        # Formats the tuples with special PAD and END characters
        for sentence in sentences:
            sentence = list(sentence)
            sentence.insert(len(sentence), 'END')
            for _ in range(self.n - 1):
                sentence.insert(0, 'PAD')
            self.uniquewords.update(sentence)

            # Creates an n-gram and an n-gram-minus1 lexicon
            for i in range(len(sentence) - self.n + 1):
                n_gram = tuple(word for word in sentence[i:i+n])
                n_gram_minus1 = tuple(word for word in sentence[i:i+n-1])

                self.n_grams[n_gram] = self.n_grams.get(n_gram, 0) + 1
                self.n_grams_minus1[n_gram_minus1] = self.n_grams_minus1.get(n_gram_minus1, 0) + 1

        self.V = len(self.uniquewords)
        self.kV = self.k * self.V

    def p(self, word:str, context:tuple[str]) -> float:
        # Returns the log probability of a word following after context of len n-1
        n_gram = context + (word, )
        logprob = math.log(self.n_grams.get(n_gram, 0) + self.k) - math.log(self.n_grams_minus1.get(context, 0) + (self.kV))

        return logprob

    def score(self, sentence: tuple[str]) -> float:
        # Returns the probability of the whole sentence calculated as the sum of the log probability
        # of each word given its n-1 context

        s_prob = 0
        sentence = list(sentence)

        # Formats the sentence with special PAD and END characters
        for _ in range(self.n - 1):
            sentence.insert(0, 'PAD')
        sentence.insert(len(sentence), 'END')
 
        for i in range(self.n - 1, len(sentence)):
            context = tuple(sentence[i - self.n + 1 : i])
            word = sentence[i]
            w_prob = self.p(word, context)
            # print(f"Prob of '{word}' given '{context}' is {w_prob}")
            s_prob += w_prob
            # print(f"sprob is {s_prob} for sentence: {sentence[:i+1]}")

        return s_prob
