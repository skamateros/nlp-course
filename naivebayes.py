# A bag-of-words Naive Bayes text classifier based on equations (4.10) and (4.14) in Jurafsky and Martin (2024, Chapter 4).

# Achieves the following accuracies on the provided datasets.
# Accuracy for imdb: 71.8%
# Accuracy for amazon: 79.6%
# Accuracy for yelp: 79.8%

from tokenizer import tokenize
from categorized_corpus import CategorizedCorpus
import numpy as np
from collections import defaultdict

def extract_sentences(cc: CategorizedCorpus) -> dict:
    '''
    Args:
        - cc (Categorized Corpus Object)

    Returns:
        - sentences (dict):
            - keys: pos/neg
            - values: defaultdict(set):
                - keys: index
                - values: set: tokens in sentence
    '''

    sentences = {'pos':defaultdict(set), 'neg':defaultdict(set)} # indexed pos/neg sentences

    index_pos = 0
    index_neg = 0
    for text, category in cc:
        tokens = tokenize(text.lower())
        if category == '1':
            sentences['pos'][index_pos].update(tokens)
            index_pos += 1
        elif category == '0':
            sentences['neg'][index_neg].update(tokens)
            index_neg += 1
    
    return sentences

def word_freq(sentences: dict) -> dict:
    '''
    Args:
        - sentences (dict):
            - keys: pos/neg
            - values: defaultdict(set):
                - keys: index
                - values: set: tokens in sentence

    Return:
        - counts (dict):
            - keys: pog/neg
            - values: times a word occurs in each category
    '''
    counts = {'pos':defaultdict(int), 'neg':defaultdict(int)} # word counts in pos/neg sentences
    for c, words in sentences.items():
        for sets in words.values():
            for word in sets:
                counts[c][word] += 1
    return counts

def train_naive_bayes(sentences: dict) -> tuple[list, dict, dict]:
    '''
    Args:
        - sentences (dict):
                - keys: pos/neg
                - values: defaultdict(set):
                    - keys: index
                    - values: set: tokens in sentence

    Returns:
        - V (list): vocabulary list
        - logprior (dict): prior probabilities per category
        - logprobs (dict): log probabilities per word per category
    '''

    # prior probabilities
    logprior = {}
    logprior['pos'] = np.log(len(sentences['pos']) / (len(sentences['pos']) + len(sentences['neg'])))
    logprior['neg'] = np.log(1 - logprior['pos'])

    counts = word_freq(sentences) # word counts in pos/neg sentences

    V = [] # vocab (just words)
    for c in counts:
        V.extend(word for word in counts[c])

    total_counts = {}
    total_counts['pos'] = sum([v for v in counts['pos'].values()]) + len(V)
    total_counts['neg'] = sum([v for v in counts['neg'].values()]) + len(V)
    
    logprobs = {'neg' : defaultdict(float), 'pos' : defaultdict(float)}
    for c in logprobs:
        for word in V:
            logprobs[c][word] = np.log((counts[c][word] + 1) / (total_counts[c]))

    return V, logprior, logprobs

def test_naive_bayes(text: list[str], logprior: dict, logprobs: dict, V: list) -> str:
    '''
    Args:
        - text (list): list of tokens in a sentence
        - logprior (dict): prior probabilities per category
        - logprobs (dict): log probabilities per word per category
        - V (list): vocabulary list

    Returns:
        - category (pos/neg)
    '''
    probs = {}
    for c in ['pos','neg']:
        prob = logprior[c]
        for token in text:
            if token in V:
                prob += logprobs[c][token]
        probs[c] = prob
    
    return max(probs.items(), key = lambda x: x[1])[0]

def main():
    
    directories = ['data/sentiment_sentences/imdb','data/sentiment_sentences/amazon','data/sentiment_sentences/yelp']

    for directory in directories:
        cc = CategorizedCorpus(directory + '/train')
        sentences = extract_sentences(cc) # indexed pos/neg sentences
        V, logprior, logprobs = train_naive_bayes(sentences)

        testcc = CategorizedCorpus(directory + '/test')
        test_sentences = extract_sentences(testcc)
        count = len(test_sentences['pos']) + len(test_sentences['neg'])
        
        n_correct = 0
        for c in test_sentences:
            for sentence in test_sentences[c].values():
                if test_naive_bayes(sentence, logprior, logprobs, V) == c:
                    n_correct += 1
        print(f'Accuracy for {directory.split('/')[-1]}: {n_correct / count * 100:.3}%')

    # while True:
    #     sentence = input('Sentence to classify:  (or exit to exit) \n')
    #     if sentence.lower() == 'exit': break

    #     X = tokenize(sentence)
    #     classification = test_naive_bayes(X, logprior, logprobs, V)

    #     print('CLASS: ', classification[0])
    

if __name__ == "__main__":
    main()