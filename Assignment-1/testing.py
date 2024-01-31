import collections, re


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = collections.defaultdict(int)
        self.tokens = []

def learn_vocabulary(self, corpus, num_merges):
    # Count character frequencies in the corpus
    word_freqs = collections.defaultdict(int)
    for word in corpus:
        # print("word: ", word)
        _word = ' '.join(list(word)) + ' </w>'
        word_freqs[_word] += 1

    # Create the initial vocabulary
    for word, freq in word_freqs.items():
        self.vocab[word] = freq

    print("self.vocab", self.vocab)
    for word in self.vocab:
        self.tokens.extend(word.split())
    # Learn the split rules and frequencies
    for _ in range(num_merges):
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            # print("symbols: ", symbols)
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        # print("pairs: ", pairs)
        if not pairs:
            break

        # Get the most frequent pair
        most_frequent_pair = max(pairs, key=pairs.get)
        # print("most_frequent_pair: ", most_frequent_pair)
        self.merges[most_frequent_pair] += 1
        # print("self.merges: ", self.merges)
        # Merge the most frequent pair in the vocabulary
        new_vocab = {}
        bigram = re.escape(' '.join(most_frequent_pair)) 
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
        for word in self.vocab:
            # new_vocab[word] = self.vocab[word]
            new_word = p.sub(''.join(most_frequent_pair), word)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab
        
        # traverse the vocab and add the new words to mastervocab
        for word in self.vocab:
            self.tokens.extend(word.split())
        
        print("self.vocab: ", self.vocab)

corpus = []
with open('corpus.txt', 'r') as f:
    for line in f:
        # print("line.strip()", line.strip())
        corpus.extend(line.strip().split())

print("corpus", corpus)

Tokenizer.learn_vocabulary = learn_vocabulary
# Tokenizer.tokenize = tokenize
# Tokenizer._split_word = _split_word
# learn vocabulary from corpus
tokenizer = Tokenizer()
tokenizer.learn_vocabulary(corpus, 9)
# print("tokenizer.vocab", tokenizer.vocab)

tokenizer.tokens = set(tokenizer.tokens)
print("tokenizer.tokens", tokenizer.tokens)
# tokens.sort()
# write tokens to tokens.txt
with open('tokens.txt', 'w') as f:
    for token in tokenizer.tokens:
        f.write(token + '\n')
# print("corpus: ", corpus)
        
# print self.merges
with open('merges.txt', 'w') as f:
    for merge in tokenizer.merges:
        f.write(merge[0] + ' ' + merge[1] + '\n')
