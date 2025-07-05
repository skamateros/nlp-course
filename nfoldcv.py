# N-fold cross-validation implemented in pure python
# The current example uses the naive bayes text classifier found in naivebayes.py

import naivebayes
from categorized_corpus import CategorizedCorpus
from collections import defaultdict

def n_fold(data_points: list, evaluation_function, n: int) -> list[float]:
    """
    Performs n-fold cross-validation on a dataset.

    Args:
        data_points (list): The dataset to be split into folds for cross-validation.
        evaluation_function: A function that takes two arguments (train, test) and returns an accuracy score.
        n (int): The number of folds to split the data into.

    Returns:
        list[float]: A list of evaluation scores, one for each fold.

    Notes:
        - The data is split into n folds by assigning each data point to a fold based on its index modulo n.
        - For each fold, the function uses it as the test set and the remaining folds as the training set.
        - The evaluation_function is called with the training and test sets, and its result is stored.
    """
    scores = []
    folds = defaultdict(list)

    for j in range(n):
        for i in range(len(data_points)):
            if i % n == j:
                folds[j].append(data_points[i])

    for u in range(len(folds)):
        test = folds[u]
        train = []
        for z in range(len(folds)):
            if z != u:
                train.extend(folds[z])
        scores.append(round(evaluation_function(train, test))) 

    return scores

def evaluation_function(training_set: list, test_set: list) -> float:
    sentences = naivebayes.extract_sentences(training_set)
    V, logprior, logprobs = naivebayes.train_naive_bayes(sentences)

    test_sentences = naivebayes.extract_sentences(test_set)

    count = len(test_sentences['pos']) + len(test_sentences['neg'])
    
    n_correct = 0
    for c in test_sentences:
        for sentence in test_sentences[c].values():
            if naivebayes.test_naive_bayes(sentence, logprior, logprobs, V) == c:
                n_correct += 1

    return n_correct / count * 100

def main():
    path = 'data/sentiment_sentences/imdb/test'
    cc = CategorizedCorpus(path)
    data_points = []
    for text, category in cc:
        data_points.append((text, category))

    scores = n_fold(data_points, evaluation_function, n = 10)
    print(scores)

if __name__ == "__main__":
    main()