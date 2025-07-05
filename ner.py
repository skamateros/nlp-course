# Named Entity Recognition model using the Structured Perceptron algorithm.
# Supports datasets in the style of the CoNLL-2003 NER dataset.
# Achieves an accuracy of 95.3% after training for 3 epochs on the english conll2003 dataset.

# Run as such:
# python3 ner.py data/conll2003/eng.train data/conll2003/eng.test

from collections import defaultdict
import sys

class Conll:
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def __iter__(self):
        with open(self.filepath) as file:
            sentence = list()
            skipped_blank = None

            for line in file:
                if line.startswith('-DOCSTART-'): # Skip -DOCSTART- line
                    skipped_blank = False
                    continue
                if len(line) <= 1:
                    if not skipped_blank: # Skip empty line after -DOCSTART-
                        skipped_blank = True
                        continue
                    else:
                        yield sentence
                        sentence.clear()
                else:
                    sentence.append(line.strip().split())
            if sentence: # Yields the final sentence
                yield sentence
                    
class StructuredPerceptron:
    def __init__(self, train: Conll):
        self.train = train
        self.weights = defaultdict(int)
        self.labels = list()

    def extract_features(self, sentence: list) -> tuple[list, defaultdict]:
        word_features = defaultdict(list)
        for i, word in enumerate(sentence):
            label = f'label: {word[3]}'
            if word[3] not in self.labels:
                self.labels.append(word[3])

            features = []
            # for x in range(1, min(4, len(word[0]))): # Word suffixes (doesn't improve the model)
            #         suffix = word[0][len(word[0])-x:]
            #         features.append((f'suffix: {suffix};', label))
            features.append((f'form: {word[0]}; pos: {word[1]};', label)) # (form: {current word}; pos: {N/JJ/VB/etc.} , label)
            features.append((f'uppercase: {word[0][0].isupper()}; firstword: {i == 0};', label)) # (uppercase (first letter): True/False; firstword: True/False; , label)ю
            # features.append((f'lol: {word[2]}', label))
            for j in range(min(-1, len(sentence[:i])), min(3, len(sentence[i:]))):
                if i+j < 0: continue
                
                # features.append((f'form: {word[0]}; pos: {word[1]};', label)) # (form: {current word}; pos: {N/JJ/VB/etc.} , label)
                if j != 0: features.append((f'relative: {j}; pos: {sentence[i+j][1]};', label)) # (relative: {-1/0/1}; pos: {previous/next word's N/JJ/VB/etc.}, label)
                if j != 0: features.append((f'relative: {j}; form: {sentence[i+j][0]};', label)) # (relative: {-1/0/1}; form: {previous/next word}, label)
                # features.append((f'uppercase: {word[0][0].isupper()}; firstword: {i == 0};', label)) # (uppercase (first letter): True/False; firstword: True/False; , label)ю

                for feature in features:
                    word_features[i].append(feature)

        return word_features

    def train_model(self, epochs: int = 3, lr: int = 1):
        for i in range(epochs):
            print(f'TRAINING EPOCH {i+1}...')
            for sentence in self.train:
                predictions = list()
                word_features = self.extract_features(sentence)
                if not word_features: continue
                for i in range(len(sentence)):
                    predictions.append(self.predict(word_features[i]))
                
                # print(f'Predictions for sequence: {' '.join(word[0] for word in sentence)} \n', predictions)

                for (i, word), prediction in zip(enumerate(sentence), predictions):
                    if word[3] != prediction:
                        for feature in word_features[i]: 
                            self.weights[feature[0], word[3]] += lr
                            self.weights[feature[0], prediction] -= lr

    def predict(self, active_features: list) -> str:
        scores = {label: 0 for label in self.labels}
        for label in self.labels:
            for feature in active_features:
                tuple = (feature[0], label)
                scores[label] += self.weights[tuple]
        return max(scores, key=scores.get)

    def test_model(self, test: Conll):
        print('TESTING MODEL...')
        correct = []
        for sentence in test:
            predictions = list()
            word_features = self.extract_features(sentence)
            if not word_features: continue
            for i in range(len(sentence)):
                predictions.append(self.predict(word_features[i]))

            for (i, word), prediction in zip(enumerate(sentence), predictions):
                    # print(f'{word[0]} | Gold: {word[3]} | Pred: {prediction}')
                    correct.append(word[3] == prediction)
  
        accuracy = (sum(correct) / len(correct)) * 100
        print(f'ACCURACY: {accuracy:0.2f}%')


def main():
    print('LOADING DATA...')
    
    train = Conll(sys.argv[1])
    test = Conll(sys.argv[2])

    model = StructuredPerceptron(train)
    model.train_model(epochs = 3)
    model.test_model(test)

    sorted_w = sorted(((w, v) for (w, v) in model.weights.items() if w[1].startswith('I-')), key=lambda x: x[1], reverse=True)[:10]
    print('-'*40)
    print('TOP 10 WEIGHTS:')
    for f, w in sorted_w:
        print(f'{f=} {w=:0.2f}')

if __name__ == '__main__':
    main()
