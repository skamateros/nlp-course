# Neural N-gram model
# Uses a word embedding layer, one linear layer and one hidden layer.
# It currently supports gzip files containing one sentence per line 
# but it can be adjusted to work with other formats.

# Run as such:
# python3 neural_ngram.py training.txt.gz

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import gzip
from tokenizer import tokenize
from tqdm import tqdm
import pickle
import os
import sys

class File:

    def __init__(self, fhandle: str):
        self.fhandle = fhandle
        self.vocab = { '<UNK>', '<PAD>', '<EOS>' }
        self.counter = Counter(word for tokens in self for word in tokens)
        self.vocab.update(word for word, count in self.counter.items() if count >= 5)
        self.word_idx = { word:i for i, word in enumerate(self.vocab) }
        if os.path.exists('vocab.pkl'):
            with open('vocab.pkl', 'rb') as f:
                self.word_idx = pickle.load(f)
                print('Vocab loaded.')
        self.idx_word = { i:word for word, i in self.word_idx.items() }
        self.V = len(self.vocab)

    def __iter__(self):
        with gzip.open(self.fhandle, 'rt') as f:
            for line in f:
                if len(line) > 0:
                    yield tokenize(line.lower())
    
class NgramDataset(Dataset):
    def __init__(self, file: File, n: int = 5):
        self.n = n
        self.data = []
        self.word_idx = file.word_idx

        unk_idx = self.word_idx['<UNK>']

        for tokens in file:
            tokens = ['<PAD>'] * (n - 1) + tokens + ['<EOS>']
            indices = [file.word_idx.get(w, file.word_idx['<UNK>']) for w in tokens]
            for i in range(len(indices) - n + 1):
                context = indices[i:i+n-1]
                target = indices[i+n-1]
                if target == unk_idx: continue
                self.data.append((context, target))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

class NeuralNgram(nn.Module):
    def __init__(self, V, context_size = 4, emb_dim = 100, hidden_dim = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=V, embedding_dim=emb_dim)
        self.linear1 = nn.Linear(in_features = emb_dim * context_size, out_features = hidden_dim, bias = True)
        self.act_fun = nn.Tanh()
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = V, bias = False)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act_fun(x)
        x = self.linear2(x)
        x = nn.functional.log_softmax(x, dim = 1)
        return x
    
def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    #train = File('data/english-train.txt.gz')
    train = File(sys.argv[1])
    model = NeuralNgram(train.V).to(device)

    dataset = NgramDataset(train, n = 5)
    dataloader = DataLoader(dataset, batch_size = 1024, shuffle = True)

    gradient_threshold = 1
    epochs = 3
    loss_func = nn.NLLLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    if os.path.exists('neural_ngram.pt'):
        model.load_state_dict(torch.load('neural_ngram.pt'))
        print('Μodel loaded.')

    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    output = 'neural_ngram.pt'
    torch.save(model.state_dict(), output)
    print(f'Saved model to {output}')

    outputdic = 'vocab.pkl'
    with open(outputdic, 'wb') as f:
        pickle.dump(train.word_idx, f)
    print(f'Exported vocabulary to {outputdic}')

if __name__ == '__main__':
    main()