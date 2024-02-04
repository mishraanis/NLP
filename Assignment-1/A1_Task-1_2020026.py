import collections
import re


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = collections.defaultdict(int)
        self.tokens = []


def learn_vocabulary(self, corpus, num_merges):
    word_freqs = collections.defaultdict(int)
    for word in corpus:
        _word = ' '.join(list(word)) + ' </w>'
        word_freqs[_word] += 1

    for word, freq in word_freqs.items():
        self.vocab[word] = freq

    for word in self.vocab:
        self.tokens.extend(word.split())

    for _ in range(num_merges):
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        if not pairs:
            break

        most_frequent_pair = max(pairs, key=pairs.get)
        self.merges[most_frequent_pair] += 1
        new_vocab = {}
        bigram = re.escape(' '.join(most_frequent_pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in self.vocab:
            new_word = p.sub(''.join(most_frequent_pair), word)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

        for word in self.vocab:
            self.tokens.extend(word.split())

def tokenize(self, sentence):
    sentence = list('$'.join(list(sentence.split())))
    sentence.append('$')
    for i in range(len(sentence)):
        if (sentence[i] == '$'):
            sentence[i] = '</w>'

    sentence = '_'+'_,_'.join(sentence) + '_'
    for merge_rules in self.merges:
        rule1 = '_'+'_,_'.join(merge_rules)+'_'
        rule2 = '_'+(''.join(merge_rules)) + '_'
        sentence = sentence.replace(rule1, rule2)
    return sentence.replace('_', '').replace('</w>', '$')


Tokenizer.learn_vocabulary = learn_vocabulary
Tokenizer.tokenize = tokenize

corpus = []
with open('../../Assignment-1/Dataset/corpus.txt', 'r') as f:
    for line in f:
        corpus.extend(line.strip().split())

tokenizer = Tokenizer()
tokenizer.learn_vocabulary(corpus, 1000)

tokens = set(tokenizer.tokens)
with open('tokens.txt', 'w') as f:
    for token in tokens:
        token = token.replace('</w>', '$')
        f.write(token + '\n')

with open('merge_rules.txt', 'w') as f:
    for merge, freq in tokenizer.merges.items():
        merge = list(merge)
        merge[0] = merge[0].replace('</w>', '$')
        merge[1] = merge[1].replace('</w>', '$')
        f.write(merge[0] + ',' + merge[1] + '\n')

sample_corpus = []
with open('sample_corpus.txt', 'r') as f:
    for line in f:
        sample_corpus.append(line.strip())

for sentence in sample_corpus:
    with open('tokenized_samples.txt', 'a') as f:
        f.write(tokenizer.tokenize(sentence) + '\n')
