from ngram_model import NGramModel
from categorized_corpus import CategorizedCorpus
from collections import defaultdict
from tokenizer import tokenize

class Classifier:
    """
    A text classification model using n-gram language models for each category.

    Args:
        train (CategorizedCorpus): The training data, as an iterable of (text, category) pairs.
        mode (str, optional): Tokenization mode, either 'word' or 'char'. Defaults to 'word'.

    Attributes:
        mode (str): Tokenization mode.
        categories (set): Set of unique category labels.
        data (defaultdict): Processed training data, grouped by category.
        models (dict): Trained n-gram models for each category.

    Methods:
        _process_data(cc):
            Processes a CategorizedCorpus into tokenized data by category.
        train_ngrams(n=2, k=0.1):
            Trains an n-gram model for each category using the processed data.
            Args:
                n (int): The order of the n-gram model. Defaults to 2.
                k (float): Smoothing parameter. Defaults to 0.1.
        test_model(test):
            Evaluates the classifier on a test CategorizedCorpus and prints accuracy.
            Args:
                test (CategorizedCorpus): The test data.
        predict(sentence):
            Predicts the category for a given tokenized sentence.
            Args:
                sentence: The input sentence, tokenized as per the mode.
            Returns:
                str: The predicted category label.
    """
    def __init__(self, train: CategorizedCorpus, mode: str = 'word'):
        self.mode = mode
        self.categories = set()
        self.data = self._process_data(train)

    def _process_data(self, cc: CategorizedCorpus) -> defaultdict:
        data = defaultdict(list)
        for text, category in cc:
            self.categories.add(category)

            if self.mode == 'char':
                data[category].append(tuple(char for char in text))
            elif self.mode == 'word':
                data[category].append(tuple(tokenize(text)))
            else:
                raise ValueError('Invalid mode: Has to be either char or word')
        return data

    def train_ngrams(self, n: int = 2, k: float = 0.1):
        self.models = {}
        for category in self.categories:
            self.models[category] = NGramModel(n = n, k = k, sentences = self.data[category])
    
    def test_model(self, test: CategorizedCorpus):
        golden = []
        predictions = []
        test_data = self._process_data(test)
        for category in self.categories:
            for data_point in test_data[category]:
                predictions.append(self.predict(data_point))
                golden.append(category)

        correct = []
        for prediction, actual in zip(predictions, golden):
            correct.append(prediction == actual)
        
        print(f'Accuracy: {sum(correct)/len(correct)*100:0.2f}%')

    def predict(self, sentence) -> str:
        scores = defaultdict(float)
        for category in self.categories:
            scores[category] = self.models[category].score(sentence)
        return max(scores, key=scores.get)  
            
def main():

    print('LANGUAGE CLASSIFICATION MODEL:')
    train = CategorizedCorpus('data/langid/langid-train.tsv')
    test = CategorizedCorpus('data/langid/langid-test.tsv')

    model = Classifier(train, mode = 'char')
    model.train_ngrams(n = 3, k = 0.1)
    model.test_model(test)

    print('SENTIMENT ANALYSIS MODEL:')
    train2 = CategorizedCorpus('data/imdb/train')
    test2 = CategorizedCorpus('data/imdb/test')

    model2 = Classifier(train2, mode = 'word')
    model2.train_ngrams(n = 2, k = 0.1)
    model2.test_model(test2)

if __name__ == "__main__":
    main()