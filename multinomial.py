# Multinomial Logistic Regression (MLR) model for morphological tag classification.
# It currently uses word affixes and POS as features, but features can be manually adjusted.
# Supports formats such as the Unimorph datasets found in the data folder.
# Achieves 66.7% accuracy after training for 3 epochs on the swedish unimorph dataset.

# Run as such:
# python3 multinomial.py data/unimorph/swe.train data/unimorph/swe.test

import csv
import numpy as np
import sys

class Unimorph:
    """
    A class for iterating over a Unimorph-formatted tab-separated file.

    Attributes:
        file (str): Path to the Unimorph file.

    Methods:
        __init__(file: str):
            Initializes the Unimorph instance with the given file path.

        __iter__():
            Yields tuples for each valid line in the file.
            Each tuple contains:
                - The lemma (with spaces replaced by underscores)
                - The first morphological tag (before the first semicolon)
                - The remaining morphological tags (after the first semicolon, joined by semicolons)
            Skips lines where the lemma or morphological tags are empty.
    """

    def __init__(self, file: str):
        self.file = file

    def __iter__(self):
        with open(self.file, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter='\t')
            for line in reader:
                if line[1].strip() and line[2].strip():
                    yield (line[1].replace(' ', '_'), line[2].split(';')[0], ';'.join(line[2].split(';')[1:]))

class MLR:

    def __init__(self, data: Unimorph):
        self.features = set()
        self.tags = set()
        self.setup(data)

    def setup(self, data: Unimorph):
        for word, pos, tag in data:
            self.tags.add(tag)
            active_features = self.extract_features(word, pos)
            self.features.update(active_features)
        
        # Mappings of tags and features to indices
        self.tag_idx = {tag:i for i, tag in enumerate(self.tags)}
        self.idx_tag = {i:tag for tag, i in self.tag_idx.items()}
        self.feature_idx = {feature:i for i, feature in enumerate(self.features)}
        self.idx_feature = {i:feature for feature, i in self.feature_idx.items()}

    def encode(self, word: str, pos: str) -> list:
        """
        Encodes a given word and its part-of-speech (POS) tag into a list of feature indices.

        This method extracts active features for the provided word and POS tag, then maps each
        active feature to its corresponding index using the feature_idx dictionary. Only features
        present in feature_idx are included in the output.

        Args:
            word (str): The word to encode.
            pos (str): The part-of-speech tag associated with the word.

        Returns:
            list: A list of indices corresponding to the active features for the given word and POS tag.
        """

        active_features = self.extract_features(word, pos)

        feature_vector = []
        for feature in active_features:
            if feature in self.feature_idx:
                feature_vector.append(self.feature_idx[feature])

        # Returns a list of indices of the active features
        return feature_vector

    def extract_features(self, word: str, pos: str) -> set:
        """
        Extracts word features.

        Args:
            word (str): A word in the data.
            pos (str): The word's pog tag.

        Returns:
            set: A set of strings where each string contains the pos tag and affix.
        """
        features = set()
        for i in range(1, min(6, len(word))):
            prefix = word[:i]
            features.add(f'{pos} {prefix=}')
            suffix = word[len(word)-i:]
            features.add(f'{pos} {suffix=}')
        return features
    
    def train(self, data: Unimorph, epochs: int = 3, lr: float = 1):
        # Format the data 
        X = []
        y = []
        for word, pos, tag in data:
            X.append(self.encode(word, pos))
            y.append(self.tag_idx[tag])
        # print(X)
        y = np.array(y)

        # Initialise weight matrix with 0s
        self.W = np.zeros(shape = (len(self.tags), len(self.features)))
        self.b = np.zeros(shape = len(self.tags))
        
        # print(f'number of features: {len(self.W[0])}')
        # print('weight matrix shape: ', self.W.shape)
        # print('n tags: ', len(self.tags))

        for epoch in range(epochs):
            print(f'EPOCH {epoch + 1}...')
            for datapoint, correct in zip(X, y):
                z =  self.score(datapoint)
                probs = self.softmax(z)

                assert round(sum(probs)) == 1 , f'Probabilities do not sum up to 1, {sum(probs)=}'

                prediction = self.predict(probs) # index of the prediction
                loss = self.loss(probs[correct])

                # print(f'loss: {loss}')

                self.update(probs, datapoint, correct, lr)
        
            # print(self.W)

    def test(self, data: Unimorph):
        total = 0
        correct = 0
        gold = []
        predicted_labels = []
        for word, pos, tag in data:
            datapoint = self.encode(word, pos)
            if not datapoint: continue
            z = self.score(datapoint)
            probs = self.softmax(z)
            prediction = self.predict(probs)
            gold.append(tag)
            predicted_labels.append(self.idx_tag[prediction])
            if self.idx_tag[prediction] == tag:
                correct += 1
            total += 1

        accuracy = correct / total * 100
        
        return accuracy, gold, predicted_labels

    def score(self, datapoint):
        z = np.zeros(shape = len(self.tags))
        for i in range(len(self.tags)):
            z[i] = sum(self.W[i, datapoint]) + self.b[i]
        return z

    def softmax(self, z):
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z))

    def predict(self, probs):
        return np.argmax(probs)
    
    def loss(self, prob):
        prob += 1e-9 if prob == 0 else 0
        return - np.log(prob)

    def update(self, probs, feature_vector, y, lr):
        for i in range(len(self.tags)):
            gradient = probs[i]
            if i == y:
                gradient -= 1  # subtract 1 for the correct class

            for feature_idx in feature_vector:
                self.W[i, feature_idx] -= lr * gradient

            self.b[i] -= lr * gradient


def main():
    train = Unimorph(sys.argv[1])
    test = Unimorph(sys.argv[2])

    model = MLR(train)
    model.train(data = train, epochs = 3, lr = 0.1)

    accuracy, gold, predicted_labels = model.test(data = test)
    print(f'Model accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()